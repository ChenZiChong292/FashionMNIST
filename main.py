from torch import nn
import torch


class Class1DModel(nn.Module):
    def __init__(self):
        super(Class1DModel, self).__init__()
        self.flatten = nn.Flatten(start_dim=2)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1)
        self.max = nn.MaxPool1d(kernel_size=3, stride=3)
        self.liner = nn.Linear(260, 1024)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1)

        self.classifier = nn.Sequential(
            nn.Linear(64 * 1022, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.conv1(x)
        x = self.max(x)
        x = self.liner(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Class2DModel(nn.Module):
    def __init__(self):
        super(Class2DModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)  # conv1
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2)  # max1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # conv2
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # conv3
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=2)  # max2
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # conv4
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # conv5
        self.relu = nn.ReLU()
        self.line1 = nn.Linear(128 * 12 * 12, 2048)
        self.line2 = nn.Linear(2048, 1024)
        self.line3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.max2(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.line1(x)
        x = self.relu(x)
        x = self.line2(x)
        x = self.relu(x)
        x = self.line3(x)
        return x


if __name__ == '__main__':
    model = Class2DModel()
    img = torch.ones([1, 3, 112, 112])
    output = model(img)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    print(1)
