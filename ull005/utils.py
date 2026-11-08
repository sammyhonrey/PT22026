"""Data loading helpers for the chest X-ray pneumonia project."""
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMG_SIZE = 224
IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
VIT_STATS = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
NORM_STATS = {"resnet50": IMAGENET_STATS, "vit": VIT_STATS}


def get_transforms(model_name):
    """Return (train_transform, eval_transform) for the given model."""
    mean, std = NORM_STATS[model_name]
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # No horizontal flip: chest anatomy is not left-right symmetric.
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def make_dataloaders(data_dir, model_name, batch_size=32,
                     val_fraction=0.2, seed=42, num_workers=2):
    """Build train/val/test loaders.

    The original val folder has only 16 images, so we re-split the
    training folder 80/20 (stratified) and keep the test set untouched.
    """
    data_dir = Path(data_dir)
    for split in ("train", "test"):
        if not (data_dir / split).is_dir():
            raise FileNotFoundError(f"Missing folder: {data_dir / split}")

    train_tf, eval_tf = get_transforms(model_name)
    full_train = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    full_val = datasets.ImageFolder(data_dir / "train", transform=eval_tf)
    test_set = datasets.ImageFolder(data_dir / "test", transform=eval_tf)

    targets = np.array(full_train.targets)
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), test_size=val_fraction,
        stratify=targets, random_state=seed)

    train_set = Subset(full_train, train_idx.tolist())
    val_set = Subset(full_val, val_idx.tolist())

    # Class weights to compensate for the imbalance (more pneumonia than normal).
    counts = np.bincount(targets[train_idx])
    class_weights = torch.tensor(
        len(train_idx) / (len(counts) * counts), dtype=torch.float)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, full_train.classes, class_weights