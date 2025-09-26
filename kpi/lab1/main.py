import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from network import Network
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from dataset import LFWIdentity
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
import csv
from collections import defaultdict

IMAGES_SUBDIR = "lfw-deepfunneled/lfw-deepfunneled"
WEIGHTS_FILE = "siamese_network.pth"
SOME_GUY = "/Users/maksym/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg"
OTHER_PHOTO_OF_THIS_GUY = "/Users/maksym/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0003.jpg"
OTHER_GUY = "/Users/maksym/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/lfw-deepfunneled/Aaron_Pena/Aaron_Pena_0001.jpg"
MATCH_TEST_CSV = "matchpairsDevTest.csv"
MISMATCH_TEST_CSV = "mismatchpairsDevTest.csv"


EPOCHS = 12

device = torch.device("mps")
model = Network().to(device)

tfm_eval = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def main():
    path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
    print(f"path: {path}")

    tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = LFWIdentity(
        root=path,
        images_subdir=IMAGES_SUBDIR,
        transform=tfm
    )

    data_loader = DataLoader(dataset, shuffle=True, batch_size=256, num_workers=0, pin_memory=True)

    loss_func = TripletMarginLoss(margin=1.0)
    mining_func = TripletMarginMiner(margin=1.0, type_of_triplets="hard")

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    losses = []
    if Path(WEIGHTS_FILE).exists():
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device, weights_only=True))
        print("Loaded pretrained weights.")
    else:
        for e in range(EPOCHS):
            running_loss = 0.0
            for imgs, labels in data_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                embeddings = model(imgs)
                hard_triplets = mining_func(embeddings, labels)

                if len(hard_triplets) > 0:
                    loss = loss_func(embeddings, labels, hard_triplets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

            scheduler.step()
            if len(data_loader) > 0:
                loss = running_loss / len(data_loader)
                losses.append(loss)
                print(f"Epoch [{e + 1}/{EPOCHS}] Loss: {loss:.4f}")
            else:
                print(f"Epoch [{e + 1}/{EPOCHS}] No batches to process.")

        torch.save(model.state_dict(), WEIGHTS_FILE)
        print("Saved model weights.")
        plt.plot(losses)
        plt.show()

    test_accuracy = evaluate(path, threshold=0.5)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    check_similarity(
        SOME_GUY,
        OTHER_GUY
    )


# A helper class for the testing dataset
class LFWTestPairs(torch.utils.data.Dataset):
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
            next(reader)
            for row in reader:
                if len(row) == 3:
                    name, n1, n2 = row
                    n1, n2 = int(n1), int(n2)
                    img1 = self.images_path / name / f"{name}_{n1:04d}.{ext}"
                    img2 = self.images_path / name / f"{name}_{n2:04d}.{ext}"
                    if img1.exists() and img2.exists():
                        self.samples.append((img1, img2, 1))
                elif len(row) == 4:
                    name1, n1, name2, n2 = row
                    n1, n2 = int(n1), int(n2)
                    img1 = self.images_path / name1 / f"{name1}_{n1:04d}.{ext}"
                    img2 = self.images_path / name2 / f"{name2}_{n2:04d}.{ext}"
                    if img1.exists() and img2.exists():
                        self.samples.append((img1, img2, 0))

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


def evaluate(dataset_path, threshold=0.5):
    test_ds_matched = LFWTestPairs(
        root=dataset_path,
        csv_subpath=MATCH_TEST_CSV,
        images_subdir=IMAGES_SUBDIR,
        label=1.0,
        transform=tfm_eval
    )
    test_ds_mismatched = LFWTestPairs(
        root=dataset_path,
        csv_subpath=MISMATCH_TEST_CSV,
        images_subdir=IMAGES_SUBDIR,
        label=0.0,
        transform=tfm_eval
    )
    test_dataset = ConcatDataset([test_ds_matched, test_ds_mismatched])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            out1 = model(x1)
            out2 = model(x2)
            cos_sim = F.cosine_similarity(out1, out2)

            predicted = (cos_sim > threshold).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = 100 * correct / total
    return accuracy


def check_similarity(img_path1: str, img_path2: str, threshold: float = 0.5):
    img1 = tfm_eval(Image.open(img_path1).convert("RGB")).unsqueeze(0).to(device)
    img2 = tfm_eval(Image.open(img_path2).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        emb1 = model(img1)
        emb2 = model(img2)

    cos_sim = F.cosine_similarity(emb1, emb2).item()
    print(f"Cosine similarity: {cos_sim:.4f}")

    if cos_sim > threshold:
        print("Faces are similar")
    else:
        print("Faces are different")


if __name__ == "__main__":
    main()