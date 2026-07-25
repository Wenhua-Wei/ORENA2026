import torch.nn as nn


class DummyModel(nn.Module):
    """Minimal CNN for verifying GPU execution. Replace with your real model."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, 2)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
