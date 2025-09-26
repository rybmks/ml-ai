import csv
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from collections import defaultdict

class LFWIdentity(TorchDataset):
    def __init__(self, root, images_subdir, ext="jpg", transform=None):
        self.root = Path(root)
        self.images_path = self.root / images_subdir
        self.transform = transform
        self.identities = defaultdict(list)
        self.samples = []
        self.ext = ext
        
        for identity_dir in self.images_path.iterdir():
            if identity_dir.is_dir():
                for img_path in identity_dir.glob(f"*.{self.ext}"):
                    self.identities[identity_dir.name].append(img_path)

        self.labels_map = {name: i for i, name in enumerate(self.identities.keys())}
        for identity, paths in self.identities.items():
            for path in paths:
                self.samples.append((path, self.labels_map[identity]))

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)