import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import clip
import copy

from utils import *
from lora import *
from datasets import build_dataset
from datasets.utils import build_data_loader
from finetune.utils import apply_lora, merge_lora, get_lora_parameters

def main():
    args = get_arguments()
    set_random_seed(args.seed)
    
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.float().eval()
    logit_scale = 100
    print(f"CLIP model '{args.backbone}' loaded.")

    print("Preparing dataset...")
    dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    
    # Define class splits for attacker and victim
    num_classes = len(dataset.classnames)
    mid_point = num_classes // 2
    ATTACKER_CLASSES = list(range(mid_point))
    VICTIM_CLASSES = [0] + list(range(mid_point, num_classes))
    print(f"Data split: {len(ATTACKER_CLASSES)} classes for attacker, {len(VICTIM_CLASSES)} for victim.")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        get_clip_normalize_transform()
    ])
    
    full_train_source = dataset.train_x_full

    # Create DataLoaders
    attacker_train_class_data = filter_data_source_by_classes(full_train_source, ATTACKER_CLASSES)
    attacker_fewshot_source = dataset.generate_fewshot_dataset(attacker_train_class_data, num_shots=args.shots)
    attacker_loader = build_data_loader(
        data_source=attacker_fewshot_source, batch_size=args.batch_size, tfm=train_transform, is_train=True, shuffle=True, num_workers=8
    )
    print(f"Attacker data loader created ({len(attacker_fewshot_source)} samples).")

    victim_train_class_data = filter_data_source_by_classes(full_train_source, VICTIM_CLASSES)
    victim_fewshot_source = dataset.generate_fewshot_dataset(victim_train_class_data, num_shots=args.shots)
    victim_train_loader = build_data_loader(
        data_source=victim_fewshot_source, batch_size=args.batch_size, tfm=train_transform, is_train=True, shuffle=True, num_workers=8
    )
    print(f"Victim training data loader created ({len(victim_fewshot_source)} samples).")
    
    victim_test_source = filter_data_source_by_classes(dataset.test, VICTIM_CLASSES)
    victim_test_loader = build_data_loader(
        data_source=victim_test_source, batch_size=256, is_train=False, tfm=preprocess, shuffle=False, num_workers=8
    )
    print(f"Victim test data loader created ({len(victim_test_source)} samples).")

    run_dormant_backdoor(args, clip_model, logit_scale, dataset, attacker_loader, victim_train_loader, victim_test_loader)

if __name__ == '__main__':
    main()