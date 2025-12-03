import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import copy

from torch.func import functional_call
from finetune.utils import apply_lora, lora_state_dict
from finetune.layers import LoRALayer

from utils import *

def create_dormant_backdoor(
    model_clean, loader, text_features_clean, logit_scale, target_class_idx, 
    args, unnormalizer, normalizer
):
    """
    Creates Dormant Backdoor using a bi-level optimization.
    """
    print("\n" + " Attacker Phase: Creating the Dormant Backdoor ".center(80, "="))
    
    delta = torch.zeros((1, 3, 224, 224), device='cuda', requires_grad=True)
    model_clean_for_stealth = copy.deepcopy(model_clean).cuda().eval()
    poison_model = copy.deepcopy(model_clean)
    poison_model.float().cuda()

    attacker_args = copy.deepcopy(args)
    attacker_args.r = args.attacker_r
    attacker_args.alpha = args.attacker_alpha
    apply_lora(attacker_args, poison_model)
    poison_model.cuda()
    
    initial_visual_params = {n: p.clone().detach() for n, p in poison_model.visual.named_parameters()}
    
    poison_optimizer = torch.optim.Adam(poison_model.visual.parameters(), lr=args.lr_poison)
    trigger_optimizer = torch.optim.Adam([delta], lr=args.lr_trigger)
    
    lora_param_names = get_lora_param_names(poison_model.visual)
    if not lora_param_names:
        raise ValueError("LoRA layers were not applied correctly; no LoRA parameters found for inner-loop simulation.")
    
    text_features_clean = text_features_clean.float()
    print(f"Target class: {target_class_idx}. Starting bi-level optimization...")

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for outer_step in range(args.outer_steps):
            poison_model.train() 
            total_loss_outer, stealth_loss_outer, utility_loss_outer, lethality_loss_outer = 0,0,0,0
            pbar = tqdm(loader, desc=f"Outer Loop [{outer_step+1}/{args.outer_steps}]")
            for i, (images, labels) in enumerate(pbar):
                images, labels = images.cuda(), labels.cuda()
                
                # Inner loop: Simulate victim's LoRA finetuning
                temp_lora_params = [p.clone() for p in get_lora_params(poison_model.visual)]
                current_full_params = dict(poison_model.visual.named_parameters())

                for _ in range(args.inner_steps):
                    params_for_inner_call = current_full_params.copy()
                    for name, param in zip(lora_param_names, temp_lora_params):
                        params_for_inner_call[name] = param

                    image_features_clean_inner = functional_call(
                        poison_model.visual, 
                        params_for_inner_call, 
                        (images,)
                    )
                    image_features_clean_inner = image_features_clean_inner / image_features_clean_inner.norm(dim=-1, keepdim=True)
                    logits_clean = logit_scale * image_features_clean_inner @ text_features_clean
                    loss_clean_tune = F.cross_entropy(logits_clean, labels)
                    
                    grads = torch.autograd.grad(loss_clean_tune, temp_lora_params, create_graph=True, allow_unused=True)
                    
                    updated_params = []
                    for param, grad in zip(temp_lora_params, grads):
                        if grad is not None:
                            updated_params.append(param - args.inner_lr * grad)
                        else:
                            updated_params.append(param)
                    temp_lora_params = updated_params
                
                # Outer loop: Calculate attack objectives
                poison_model.eval()
                perturbed_images = normalizer(torch.clamp(unnormalizer(images) + delta, 0.0, 1.0))
                target_labels = torch.full((images.size(0),), target_class_idx, device='cuda', dtype=torch.long)
                
                params_after_tune = current_full_params.copy()
                for name, param in zip(lora_param_names, temp_lora_params):
                    params_after_tune[name] = param
                
                image_features_lethal = functional_call(poison_model.visual, params_after_tune, (perturbed_images,))
                logits_lethal = logit_scale * (image_features_lethal / image_features_lethal.norm(dim=-1, keepdim=True)) @ text_features_clean
                loss_lethality = F.cross_entropy(logits_lethal, target_labels)
                
                image_features_utility = functional_call(poison_model.visual, params_after_tune, (images,))
                logits_utility = logit_scale * (image_features_utility / image_features_utility.norm(dim=-1, keepdim=True)) @ text_features_clean
                loss_utility = F.cross_entropy(logits_utility, labels)
                
                with torch.no_grad():
                    clean_features_on_perturbed = model_clean_for_stealth.encode_image(perturbed_images)
                    logits_clean_on_perturbed = logit_scale * (clean_features_on_perturbed / clean_features_on_perturbed.norm(dim=-1, keepdim=True)) @ text_features_clean
                
                poison_features_on_perturbed = poison_model.encode_image(perturbed_images)
                logits_poison_on_perturbed = logit_scale * (poison_features_on_perturbed / poison_features_on_perturbed.norm(dim=-1, keepdim=True)) @ text_features_clean
                
                loss_stealth = F.kl_div(
                    F.log_softmax(logits_poison_on_perturbed, dim=1),
                    F.softmax(logits_clean_on_perturbed, dim=1),
                    reduction='batchmean'
                )
                
                total_loss = loss_lethality + args.lambda_stealth * loss_stealth + args.lambda_utility * loss_utility
                
                poison_model.train()
                poison_optimizer.zero_grad()
                trigger_optimizer.zero_grad()
                total_loss.backward()
                poison_optimizer.step()
                trigger_optimizer.step()

                with torch.no_grad():
                    delta.clamp_(-args.epsilon, args.epsilon)
                
                total_loss_outer += total_loss.item(); lethality_loss_outer += loss_lethality.item(); stealth_loss_outer += loss_stealth.item(); utility_loss_outer += loss_utility.item()
                pbar.set_postfix(L_total=total_loss.item(), L_lethal=loss_lethality.item(), L_stealth=loss_stealth.item(), L_utility=loss_utility.item())

            print(f"Outer Loop {outer_step+1} Avg Loss: Total={total_loss_outer/len(loader):.4f}, Lethal={lethality_loss_outer/len(loader):.4f}, Stealth={stealth_loss_outer/len(loader):.4f}, Utility={utility_loss_outer/len(loader):.4f}")

    print("Dormant backdoor created successfully. Calculating weight delta...")
    poison_model.eval()
    poisoned_weights_delta = {
        n: p.data - initial_visual_params[n] 
        for n, p in poison_model.visual.named_parameters() 
        if n in initial_visual_params
    }
    return poisoned_weights_delta, delta.detach()


def run_dormant_backdoor(args, clip_model, logit_scale, dataset, attacker_loader, victim_train_loader, victim_test_loader):
    print("\n===== Dormant Backdoor Experiment (Full Parameter Perturbation) =====")
    
    unnormalizer = UnNormalize(mean=CLIP_NORMALIZE_MEAN, std=CLIP_NORMALIZE_STD)
    normalizer = get_clip_normalize_transform()
    target_class_idx = 0 
    print(f"Attack target: class {target_class_idx} ('{dataset.classnames[target_class_idx].replace('_', ' ')}')")
    
    with torch.no_grad():
        clean_text_features_for_eval = clip_classifier(dataset.classnames, dataset.template, clip_model)

    poisoned_weights_delta, backdoor_trigger = create_dormant_backdoor(
        model_clean=clip_model, loader=attacker_loader, text_features_clean=clean_text_features_for_eval,
        logit_scale=logit_scale, target_class_idx=target_class_idx, args=args,
        unnormalizer=unnormalizer, normalizer=normalizer
    )

    print("\nAttacker: Creating backdoored model by applying weight delta to a clean model...")
    model_poisoned_for_distribution, _ = clip.load(args.backbone)
    model_poisoned_for_distribution.float().cuda()
    
    with torch.no_grad():
        for name, param in model_poisoned_for_distribution.visual.named_parameters():
            if name in poisoned_weights_delta:
                param.add_(poisoned_weights_delta[name].to(param.device))
    
    model_poisoned_for_distribution.eval()
    print("Attacker: Poisoned model is ready for distribution.")

    print("\n===== Victim Simulation =====")
    model_victim_finetuning = copy.deepcopy(model_poisoned_for_distribution)

    print("\n--- Stealthiness Check ---")
    evaluate_attack_effectiveness(
        model=model_victim_finetuning, loader=victim_test_loader, perturbation=backdoor_trigger, 
        text_features=clean_text_features_for_eval, logit_scale=logit_scale, target_class_idx=target_class_idx, 
        description="Attack on received model before finetuning",
        unnormalizer=unnormalizer, normalizer=normalizer
    )

    print("\n--- Victim Finetuning ---")
    victim_args = copy.deepcopy(args)
    victim_args.r = args.victim_r
    victim_args.alpha = args.victim_alpha
    apply_lora(victim_args, model_victim_finetuning)
    model_victim_finetuning.cuda()

    print(f"Victim: Starting LoRA finetuning for {args.epochs} epochs...")
    optimizer = torch.optim.AdamW(get_lora_parameters(model_victim_finetuning.visual), lr=args.lr)
    
    for epoch in range(args.epochs):
        model_victim_finetuning.train()
        pbar_victim = tqdm(victim_train_loader, desc=f"Victim Finetuning Epoch {epoch+1}/{args.epochs}")
        for i, (images, target) in enumerate(pbar_victim):
            images, target = images.cuda(), target.cuda()
            image_features = model_victim_finetuning.encode_image(images).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_victim = clean_text_features_for_eval.float()
            cosine_similarity = logit_scale * image_features @ text_features_victim
            loss = F.cross_entropy(cosine_similarity, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar_victim.set_postfix(loss=loss.item())

    print("Finetuning complete.")
    model_victim_finetuning.eval()
    
    print("\n--- Final Model Evaluation ---")
    with torch.no_grad():
        final_text_features = clip_classifier(dataset.classnames, dataset.template, model_victim_finetuning)

    final_clean_acc = evaluate_lora(args, model_victim_finetuning, victim_test_loader, dataset)
    print(f"Final clean data accuracy: {final_clean_acc:.2f}%")
    
    print("\nEvaluating Lethality (Attack Success Rate):")
    evaluate_attack_effectiveness(
        model=model_victim_finetuning, loader=victim_test_loader, perturbation=backdoor_trigger, 
        text_features=final_text_features, logit_scale=logit_scale, target_class_idx=target_class_idx, 
        description="Attack on finetuned model",
        unnormalizer=unnormalizer, normalizer=normalizer
    )

    print("\n===== Experiment Finished =====\n")