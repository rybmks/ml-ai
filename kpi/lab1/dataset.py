import csv
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

class LFWPairs(TorchDataset):
    def __init__(self, root, csv_subpath, images_subdir, ext="jpg", transform=None, label=None):
        self.root = Path(root)
        self.csv_path = self.root / csv_subpath
        self.images_path = self.root / images_subdir
        self.ext = ext
        self.samples = []
        self.transform = transform
        self.label = label

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

            if len(header) == 3:  
                for row in reader:
                    name, n1, n2 = row
                    n1, n2 = int(n1), int(n2)
                    img1 = self.images_path / name / f"{name}_{n1:04d}.{ext}"
                    img2 = self.images_path / name / f"{name}_{n2:04d}.{ext}"
                    if img1.exists() and img2.exists():
                        self.samples.append((img1, img2, 1)) 

            elif len(header) == 4:
                for row in reader:
                    name1, n1, name2, n2 = row
                    n1, n2 = int(n1), int(n2)
                    img1 = self.images_path / name1 / f"{name1}_{n1:04d}.{ext}"
                    img2 = self.images_path / name2 / f"{name2}_{n2:04d}.{ext}"
                    if img1.exists() and img2.exists():
                        self.samples.append((img1, img2, 0))

            else:
                raise RuntimeError(f"Unexpected CSV format in {self.csv_path}")

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.csv_path}")

    def __getitem__(self, idx):
        p1, p2, label = self.samples[idx]
        x1 = self._open(p1)
        x2 = self._open(p2)
        y = torch.tensor(label, dtype=torch.float32)
        return x1, x2, y

    def _open(self, path: Path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
