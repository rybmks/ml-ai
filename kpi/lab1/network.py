import torch
import torch.nn as nn
import torch.nn.functional as F

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

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            out = self._forward_conv(dummy)
            flatten_size = out.numel()

        self.fc1 = nn.Linear(flatten_size, 1024)
        self.fc2 = nn.Linear(1024, 512)

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.drop(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.drop(x)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.drop(x)
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.drop(x)
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
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