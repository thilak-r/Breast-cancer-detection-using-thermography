import torch
import torch.nn as nn

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Define your CNN layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Added conv3 layer
        self.bn1 = nn.BatchNorm2d(16)  # Batch Normalization after conv1
        self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization after conv2
        self.bn3 = nn.BatchNorm2d(64)  # Batch Normalization after conv3
        self.fc1 = nn.Linear(64 * 20 * 15, 512)  # Assuming image size is 160x120
        self.fc2 = nn.Linear(512, 1)  # Binary output (Normal or Sick)

    def forward(self, x):
        # Define the forward pass with Batch Normalization and ReLU activations
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x
