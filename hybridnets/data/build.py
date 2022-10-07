import torch
from torchvision import transforms

from hybridnets.data.dataset import BddDataset

def build_transform(params, split="train"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=params.mean, std=params.std
        )
    ])
    return transform

def build_data_loader(params, split="train"):
    is_train = split == "train"
    shuffle = split == "train"
    transform = build_transform(params, split=split)

    dataset = BddDataset(
        params=params,
        is_train=is_train,
        inputsize=params.model['image_size'],
        transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=params.num_workers,
        collate_fn=BddDataset.collate_fn
    )
    return data_loader
