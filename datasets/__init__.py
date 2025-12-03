from .imagenet import ImageNet


dataset_list = {
                "imagenet": ImageNet,
                }


def build_dataset(dataset, root_path, shots, preprocess):
    full_dataset = dataset_list[dataset](root=root_path, num_shots=-1)
    full_train_data = full_dataset.train_x

    print(f"Loading {shots}-shot dataset for LoRA fine-tuning...")
    fewshot_dataset = dataset_list[dataset](root=root_path, num_shots=shots)

    fewshot_dataset._train_x_full = full_train_data
    
    return fewshot_dataset