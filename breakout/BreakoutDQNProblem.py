from DQNProblem import DQNProblem
from breakout.BreakoutDQN import BreakoutDQN
from Agent import Agent

import numpy as np
import torch
import torchvision.transforms as T
import gym
from gym.wrappers import FrameStack


class BreakoutDQNProblem(DQNProblem):
    inputHeight = 84
    inputWidth = 84
    outputSize = 4

    def __init__(self, lr, gamma, experienceReplay, epsilonDecay, device, targetUpdateFrequency):
        super().__init__(lr, gamma, experienceReplay, epsilonDecay, device, targetUpdateFrequency)

        self.environment = FrameStack(gym.make('BreakoutNoFrameskip-v4'), 4)
        self.environment = gym.wrappers.Monitor(self.environment, "recording", force=True, video_callable=lambda x: x % 100 == 0)

        self.policyNet = BreakoutDQN(self.inputWidth, self.inputHeight, self.outputSize).to(self.device)
        self.targetNet = BreakoutDQN(self.inputWidth, self.inputHeight, self.outputSize).to(self.device)
        self._initNeuralNets()

        self.agent = Agent(self.policyNet, self.epsilonDecay, self.environment.action_space.n, self.device)
        self.lastLife = 5

    def reset(self):
        super().reset()
        self.lastLife = 5

    def processTerminalState(self, done, info):
        if info["ale.lives"] < self.lastLife:
            self.lastLife = info["ale.lives"]
            return True
        return done

    def processState(self, state):
        processedScreens = np.ndarray((len(state), self.inputWidth, self.inputHeight))

        for (idx, screen) in enumerate(state):
            screen = screen.transpose((2, 0, 1))
            screen_height = screen.shape[1]
            top = int(screen_height * 0.1)
            bottom = int(screen_height * 0.93)
            screen = screen[:, top:bottom, :]
            screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
            screen = torch.from_numpy(screen)
            resize = T.Compose([
                T.ToPILImage(),
                T.Resize((self.inputWidth, self.inputHeight)),
                T.Grayscale(),
                T.ToTensor()
            ])
            processedScreens[idx] = resize(screen).squeeze(0)

        processedScreens = np.ascontiguousarray(processedScreens, dtype=np.float32)
        return torch.from_numpy(processedScreens).unsqueeze(0).to(self.device)

    def getTargetNetwork(self):
        return self.targetNet

    def getPolicyNetwork(self):
        return self.policyNet

    def getEnvironment(self):
        return self.environment

    def getAgent(self):
        return self.agent
