# resnet50_model.py
import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Model(nn.Module):
    def __init__(self, output_dim=1, pretrained=True, freeze_backbone=True):
        super(ResNet50Model, self).__init__()
        self.base_model = models.resnet50(pretrained=pretrained)

        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.base_model(x)
