import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.15)
        self.flatten = nn.Flatten()
        
        # The corrected input size for the linear layer
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 128 -> 64
        x = self.drop(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 64 -> 32
        x = self.drop(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # 32 -> 16
        x = self.drop(x)
        x = self.pool(F.relu(self.bn4(self.conv4(x)))) # 16 -> 8
        x = self.drop(x)
        x = self.pool(F.relu(self.bn5(self.conv5(x)))) # 8 -> 4
        x = self.drop(x)
        x = self.pool(F.relu(self.bn6(self.conv6(x)))) # 4 -> 2
        x = self.drop(x)
        return x

    def forward_once(self, x):
        x = self._forward_conv(x)
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x, y):
        return self.forward_once(x), self.forward_once(y)
    
    def check_similarity(self, img_path1: str, img_path2: str, threshold: float = 0.5, device = torch.device("mps")):
        self.eval()
        tfm_eval = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        img1 = tfm_eval(Image.open(img_path1).convert("RGB")).unsqueeze(0).to(device)
        img2 = tfm_eval(Image.open(img_path2).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            emb1 = self.forward_once(img1)
            emb2 = self.forward_once(img2)

        cos_sim = F.cosine_similarity(emb1, emb2).item()
        print(f"Cosine similarity: {cos_sim:.4f}")

        if cos_sim > threshold:
            print("Faces are similar")
        else:
            print("Faces are different")

    def test_model(self, data_loader, device, threshold=0.5):
        self.eval()
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for x1, x2, y in data_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                
                emb1, emb2 = self(x1, x2)
                cos_sim = F.cosine_similarity(emb1, emb2)
                
                predictions = (cos_sim > threshold).float()
                
                correct_predictions += (predictions == y).sum().item()
                total_samples += y.size(0)
        
        accuracy = correct_predictions / total_samples
        print(f"Test Accuracy: {accuracy:.4f} with threshold {threshold:.2f}")
        return accuracy