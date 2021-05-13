import torch.nn as nn

class RecommendationsDQN(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.fc1 = nn.Linear(inputSize, 256)
        self.av1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.av2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.av3 = nn.ReLU()
        self.fc4 = nn.Linear(64, outputSize)

    def forward(self, x):
        x = self.fc1(x)
        x = self.av1(x)

        x = self.fc2(x)
        x = self.av2(x)

        x = self.fc3(x)
        x = self.av3(x)

        return self.fc4(x)