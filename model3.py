import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel3(nn.Module):
    def __init__(self):
        super(CNNModel3, self).__init__()

        # Shared convolutional feature extractor
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),

            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),

            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            nn.Conv2d(48, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Flatten layer
        self.flatten = nn.Flatten()

        # Steering branch
        self.steer_head = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50,10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

        # Throttle branch
        self.throttle_head = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50,10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)

        steering = self.steer_head(x)
        throttle = self.throttle_head(x)

        return torch.cat([steering, throttle], dim=1)  # shape: [batch, 2]