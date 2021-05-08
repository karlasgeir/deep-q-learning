import numpy as np
import torch
import random


class Agent:
    currentStep = 0

    def __init__(self, nn, epsilonDecay, actionSpaceSize, device):
        self.nn = nn
        self.epsilonDecay = epsilonDecay
        self.actionSpaceSize = actionSpaceSize
        self.device = device

    def action(self, state):
        epsilon = self.epsilonDecay.step()
        self.currentStep += 1
        if np.random.rand() < epsilon:
            return torch.tensor([[random.randrange(self.actionSpaceSize)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.nn(state).max(1)[1].view(1, 1)
