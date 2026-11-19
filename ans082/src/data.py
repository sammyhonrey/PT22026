"""Datasets, transforms and data loaders for SmartSort AI."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMAGE_SIZE = 224

# Fixed class order. The index of a class in this list is its numeric label everywhere
# (training, evaluation, the app), so never reorder it.
CLASSES = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}


def get_normalization(model_name: str = "google/vit-base-patch16-224"):
    """Return (mean, std) from a Hugging Face model's own image processor."""
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(model_name)
    return list(processor.image_mean), list(processor.image_std)


def build_transforms(mean, std, image_size: int = IMAGE_SIZE, augmentation: str = "standard"):
    """Return (train_transform, eval_transform) for the given normalisation values.

    augmentation="standard" is the default recipe used for E1 to E3.
    augmentation="strong" adds perspective, blur, grayscale, stronger colour jitter and random
    erasing, to imitate real phone photos (experiment E4).
    """
    if augmentation == "standard":
        train_steps = [
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    elif augmentation == "strong":
        train_steps = [
            T.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=30),
            T.RandomPerspective(distortion_scale=0.2, p=0.3),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15), value=0),
        ]
    else:
        raise ValueError(f"Unknown augmentation: {augmentation!r} (use 'standard' or 'strong')")
    eval_steps = [
        T.Resize(int(image_size * 256 / 224)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    return T.Compose(train_steps), T.Compose(eval_steps)


def load_rgb(path) -> Image.Image:
    """Open an image and convert it to RGB (some dataset images are palette mode)."""
    with Image.open(path) as img:
        return img.convert("RGB")


class SmartSortDataset(Dataset):
    """Images of one split (train, val or test) listed in the split CSV.

    The CSV has the columns path, label, split, orig_class. Paths are relative to data_root.
    """

    def __init__(self, split_csv, data_root, split, transform=None,
                 max_samples=None, seed: int = 42):
        table = pd.read_csv(split_csv)
        table = table[table["split"] == split].reset_index(drop=True)
        if max_samples is not None and max_samples < len(table):
            table = table.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        unknown = set(table["label"]) - set(CLASSES)
        if unknown:
            raise ValueError(f"Unknown labels in split file: {sorted(unknown)}")
        if len(table) == 0:
            raise ValueError(f"No images found for split '{split}'")
        self.paths = [Path(data_root) / p for p in table["path"]]
        self.labels = [CLASS_TO_IDX[label] for label in table["label"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        image = load_rgb(self.paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def _seed_worker(worker_id: int) -> None:
    """Give every data-loading worker its own reproducible random seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 2,
                 seed: int = 42) -> DataLoader:
    """Create a DataLoader. For training (shuffle=True) the last partial batch is dropped."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )
