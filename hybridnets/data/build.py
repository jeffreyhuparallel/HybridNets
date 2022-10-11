import torch
from torchvision import transforms

from hybridnets.data.bdd_dataset import BddDataset


def build_transform(params, split="train"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=params.mean, std=params.std
        )
    ])
    return transform

def build_dataset(params, split="train"):
    transform = build_transform(params, split=split)

    is_train = split == "train"
    dataset = BddDataset(
        params=params,
        is_train=is_train,
        transform=transform,
    )
    return dataset

def build_data_loader(params, split="train"):
    dataset = build_dataset(params, split=split)

    batch_size = params.batch_size
    num_workers = params.num_workers
    if split == "train":
        shuffle = True
    else:
        shuffle = False

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=BddDataset.collate_fn
    )
    return data_loader
