import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 20, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(20, 10, kernel_size=3, padding=1)
        self.fc = nn.Linear(10 * 7 * 7, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(-1, 10 * 7 * 7)
        x = self.fc(x)
        return x 