import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel2(nn.Module):
    def __init__(self):
        super(CNNModel2, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),  # Output: (24, 31, 98)
            nn.BatchNorm2d(24),
            nn.ReLU(),

            nn.Conv2d(24, 36, kernel_size=5, stride=2), # (36, 14, 47)
            nn.BatchNorm2d(36),
            nn.ReLU(),

            nn.Conv2d(36, 48, kernel_size=5, stride=2), # (48, 5, 22)
            nn.BatchNorm2d(48),
            nn.ReLU(),

            nn.Conv2d(48, 64, kernel_size=3),           # (64, 3, 20)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3),           # (64, 1, 18)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 100),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(50, 10),
            nn.ReLU(),

            nn.Linear(10, 2)  # Output: steering angle + throttle
        )
    
    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x
