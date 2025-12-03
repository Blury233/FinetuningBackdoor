import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from tqdm import tqdm
import torchvision.transforms as transforms

CLIP_NORMALIZE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_NORMALIZE_STD = (0.26862954, 0.26130258, 0.27577711)

class UnNormalize(object):
    """Reverses the normalization on a tensor of images."""
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    def __call__(self, tensor):
        std_dev = self.std.to(tensor.device)
        mean_dev = self.mean.to(tensor.device)
        return tensor * std_dev + mean_dev

def get_clip_normalize_transform():
    """Returns the normalization transform for CLIP."""
    return transforms.Normalize(mean=CLIP_NORMALIZE_MEAN, std=CLIP_NORMALIZE_STD)

def set_random_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_lora_params(root_model):
    """Gets all LoRA parameters from a model."""
    return [p for n, p in root_model.named_parameters() if 'lora_' in n and p.requires_grad]

def get_lora_param_names(root_model):
    """Gets the names of all LoRA parameters from a model."""
    return [n for n, p in root_model.named_parameters() if 'lora_' in n and p.requires_grad]

def cls_acc(output, target, topk):
    """Computes the classification accuracy."""
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def clip_classifier(classnames, template, clip_model):
    """Computes text features for a list of classnames using a CLIP model."""
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def evaluate_lora(args, clip_model, loader, dataset):
    """Evaluates the clean accuracy of a LoRA-equipped model."""
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0]
        texts = [template.format(c.replace('_', ' ')) for c in dataset.classnames]
        texts = clip.tokenize(texts).cuda()
        class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            if image_features.shape[1] != text_features.shape[0]:
                text_features = text_features.t()
            cosine_similarity = image_features @ text_features
            acc += cls_acc(cosine_similarity, target, topk=1) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples
    return acc

def evaluate_attack_effectiveness(model, loader, perturbation, text_features, logit_scale, target_class_idx, description, unnormalizer, normalizer):
    """Evaluates the attack success rate (ASR) and accuracy on poisoned data."""
    print(f"\n--- Attack Test: {description} ---")
    model.eval()
    device = next(model.parameters()).device
    model.to(device)
    total, correct, attack_success = 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing attack..."):
            images, labels = images.cuda(), labels.cuda()
            perturbed_images = normalizer(torch.clamp(unnormalizer(images) + perturbation, 0.0, 1.0))

            image_features = model.encode_image(perturbed_images).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            text_features_eval = text_features.float()
            if image_features.shape[1] != text_features_eval.shape[0]:
                text_features_eval = text_features_eval.t()
            
            logits = logit_scale * image_features @ text_features_eval
            preds = torch.argmax(logits, dim=1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            attack_success += (preds == target_class_idx).sum().item()
            
    accuracy = 100 * correct / total
    attack_success_rate = 100 * attack_success / total
    
    print(f"Results for '{description}':")
    print(f"  -> Accuracy on poisoned data: {accuracy:.2f}%")
    print(f"  -> Attack Success Rate (ASR) for target class {target_class_idx}: {attack_success_rate:.2f}%")
    return accuracy, attack_success_rate

def filter_data_source_by_classes(data_source, keep_classes):
    """Filters a list of Datum objects to include only specified classes."""
    return [item for item in data_source if item.label in keep_classes]

def get_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()

    # Core arguments
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--shots', type=int, required=True)
    parser.add_argument('--backbone', type=str, required=True, choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'], help='CLIP backbone to use')

    # Victim Finetuning arguments
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs for victim finetuning')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate for victim finetuning')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--victim_r', type=int, required=True, help='The rank for the victim\'s LoRA')
    parser.add_argument('--victim_alpha', type=int, required=True, help='The alpha for the victim\'s LoRA')

    # Dormant Backdoor Attack arguments
    parser.add_argument('--outer_steps', type=int, required=True, help='Outer loop steps for backdoor generation')
    parser.add_argument('--inner_steps', type=int, required=True, help='Inner loop steps for simulating victim finetuning')
    parser.add_argument('--lr_poison', type=float, required=True, help='Learning rate for poisoning LoRA weights')
    parser.add_argument('--lr_trigger', type=float, required=True, help='Learning rate for the trigger pattern')
    parser.add_argument('--inner_lr', type=float, required=True, help='Learning rate for inner loop optimization')
    parser.add_argument('--lambda_stealth', type=float, required=True, help='Weight for the stealth loss')
    parser.add_argument('--lambda_utility', type=float, required=True, help='Weight for the utility loss')
    parser.add_argument('--epsilon', type=float, required=True, help='Perturbation budget for the trigger')
    parser.add_argument('--attacker_r', type=int, required=True, help='The rank for the attacker\'s LoRA')
    parser.add_argument('--attacker_alpha', type=int, required=True, help='The alpha for the attacker\'s LoRA')

    # LoRA generic arguments
    parser.add_argument('--position', type=str, required=True, choices=['bottom', 'mid', 'up', 'all'], help='Where to inject LoRA modules in the vision encoder')
    parser.add_argument('--encoder', type=str, required=True, help='Which encoder to apply LoRA to')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='Attention matrices to apply LoRA')
    parser.add_argument('--dropout_rate', type=float, required=True, help='Dropout rate for LoRA')
    
    args = parser.parse_args()
    return args