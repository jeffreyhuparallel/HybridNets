import torch
from torchvision import transforms

from hybridnets.data.bdd_dataset import BddDataset


def build_transform(cfg, split="train"):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform

def build_dataset(cfg, split="train"):
    transform = build_transform(cfg, split=split)

    is_train = split == "train"
    dataset = BddDataset(
        cfg,
        is_train=is_train,
        transform=transform,
    )
    return dataset

def build_data_loader(cfg, split="train"):
    dataset = build_dataset(cfg, split=split)

    batch_size = cfg.DATALOADER.BATCH_SIZE
    num_workers = cfg.DATALOADER.NUM_WORKERS
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
