#/Users/maksym/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/lfw-deepfunneled
#/Users/maksym/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/matchpairsDevTest.csv
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from network import Network
from torchvision import transforms
from torch.utils.data import ConcatDataset, DataLoader
from dataset import LFWPairs
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np 
from helpers import find_optimal_threshold

MATCH_TRAIN_CSV = "matchpairsDevTrain.csv"
MISMATCH_TRAIN_CSV = "mismatchpairsDevTrain.csv"
MATCH_TEST_CSV = "matchpairsDevTest.csv"
MISMATCH_TEST_CSV = "mismatchpairsDevTest.csv"
IMAGES_SUBDIR = "lfw-deepfunneled/lfw-deepfunneled"
WEIGHTS_FILE = "siamese_network.pth"
SOME_GUY="/Users/maksym/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg"
OTHER_PHOTO_OF_THIS_GUY="/Users/maksym/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0003.jpg"
OTHER_GUY="/Users/maksym/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/lfw-deepfunneled/Aaron_Pena/Aaron_Pena_0001.jpg"

EPOCHS = 12

tfm_eval = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
device = torch.device("mps")
model = Network().to(device)

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
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    matched_ds = LFWPairs(
        root=path, 
        csv_subpath=MATCH_TRAIN_CSV, 
        images_subdir=IMAGES_SUBDIR,
        label=1.0,
        transform=tfm
    )
    mismatched_ds = LFWPairs(
        root=path,
        csv_subpath=MISMATCH_TRAIN_CSV,
        images_subdir=IMAGES_SUBDIR,
        label=0.0,
        transform=tfm
    )
    full_dataset = ConcatDataset([matched_ds, mismatched_ds])
    data_loader = DataLoader(full_dataset, shuffle=True, batch_size=64, num_workers=0, pin_memory=True)

    matched_test_ds = LFWPairs(
        root=path, 
        csv_subpath=MATCH_TEST_CSV, 
        images_subdir=IMAGES_SUBDIR,
        label=1.0,
        transform=tfm_eval
    )
    mismatched_test_ds = LFWPairs(
        root=path,
        csv_subpath=MISMATCH_TEST_CSV,
        images_subdir=IMAGES_SUBDIR,
        label=0.0,
        transform=tfm_eval
    )
    full_test_dataset = ConcatDataset([matched_test_ds, mismatched_test_ds])
    test_loader = DataLoader(full_test_dataset, shuffle=False, batch_size=64, num_workers=0, pin_memory=True)

    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    losses = []
    if Path(WEIGHTS_FILE).exists():
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device, weights_only=True))
        print("Loaded pretrained weights.")
    else:
        for e in range(EPOCHS):
            running_loss = 0.0
            for x1, x2, y in data_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                y = y * 2 - 1

                out1, out2 = model(x1, x2)
                loss = criterion(out1, out2, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            scheduler.step()
            loss = running_loss/len(data_loader)
            losses.append(loss)
            print(f"Epoch [{e+1}/{EPOCHS}] Loss: {loss:.4f}")

        torch.save(model.state_dict(), WEIGHTS_FILE)
        print("Saved model weights.")

    plt.plot(losses)
    optimal_threshold = find_optimal_threshold(model, test_loader, device)
    
    model.check_similarity(
        SOME_GUY,
        OTHER_GUY,
        threshold=optimal_threshold
    )

if __name__ == "__main__":
    main()