import torch
from torch import nn


class Classification2DModel(nn.Module):
    def __init__(self):
        super(Classification2DModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),  # conv1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # max1
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # conv2
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # conv3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # max2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # conv4
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # conv5
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*12*12, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Classification1DModel(nn.Module):
    def __init__(self):
        super(Classification1DModel, self).__init__()
        self.flatten = nn.Flatten(start_dim=2)
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Linear(260, 1024),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 1022, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
