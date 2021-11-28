from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
import pytorch_lightning as pl

ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.augmentations import *
from utils.transforms import DEFAULT_TRANSFORMS
from utils.utils import worker_seed_set


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class Dataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boxes = np.loadtxt(label_path).reshape(-1, 5)

        if self.transform:
            img, bb_targets = self.transform((img, boxes))

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)


class DataModule(pl.LightningDataModule):
    def __init__(self, data_config, batch_size, img_size):
        super().__init__()
        self.train_path = data_config["train"]
        self.valid_path = data_config["valid"]
        self.batch_size = batch_size
        self.img_size = img_size

    def train_dataloader(self):
        dataset = Dataset(
            self.train_path,
            img_size=self.img_size,
            transform=AUGMENTATION_TRANSFORMS)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            worker_init_fn=worker_seed_set)
        return dataloader

    def val_dataloader(self):
        dataset = Dataset(
            self.valid_path,
            img_size=self.img_size,
            transform=DEFAULT_TRANSFORMS)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            collate_fn=dataset.collate_fn)
        return dataloader
