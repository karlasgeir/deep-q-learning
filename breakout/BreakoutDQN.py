import torch
import torch.nn as nn


class BreakoutDQN(nn.Module):
    def __init__(self, width, height, outputSize):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        torch.nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 8, 4), 4, 2), 3, 2)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 8, 4), 4, 2), 3, 2)
        linear_input_size = convw * convh * 64

        self.head = nn.Sequential(nn.Linear(linear_input_size, 512), nn.ReLU(), nn.Linear(512, outputSize))

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))