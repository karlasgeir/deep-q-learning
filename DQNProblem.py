import torch
import torch.nn as nn
import torch.optim as optim

from TargetNetUpdater import TargetNetUpdater


class DQNProblem:
    currentState = None

    def __init__(self, lr, gamma, experienceReplay, epsilonDecay, device, targetUpdateFrequency):
        self.lr = lr
        self.gamma = gamma
        self.experienceReplay = experienceReplay
        self.device = device
        self.epsilonDecay = epsilonDecay
        self.targetUpdateFrequency = targetUpdateFrequency

    def _initNeuralNets(self):
        targetNet = self.getTargetNetwork()
        policyNet = self.getPolicyNetwork()
        self.targetNetUpdater = TargetNetUpdater(targetNet, policyNet, self.targetUpdateFrequency)
        # Load the parameters from the policy net to make sure they are
        # initialized with the same parameters
        targetNet.load_state_dict(policyNet.state_dict())
        # Put the target net to evaluation mode, we will manually update it
        targetNet.eval()
        # Put the policy net into training mode
        policyNet.train()

        self.optimizer = optim.Adam(params=policyNet.parameters(), lr=self.lr)

    def processState(self, state):
        return state

    def reset(self):
        self.currentState = self.processState(self.getEnvironment().reset())

    def step(self):
        action = self.getAgent().action(self.currentState)

        (nextUnprocessedState, reward, done, info) = self.getEnvironment().step(action.item())

        isTerminal = self.processTerminalState(done, info)

        nextState = self.processState(nextUnprocessedState)

        self.experienceReplay.addExperience(
            (self.currentState, action, torch.tensor([reward]).to(self.device), nextState, isTerminal))

        self.currentState = nextState

        self.updateQValues()

        self.targetNetUpdater.step()

        return reward, done

    def processTerminalState(self, done, info):
        return done

    def updateQValues(self):
        experiences = self.experienceReplay.getExperiences()
        if experiences:
            (states, actions, rewards, nextStates, isDones) = experiences

            currentQValue = self.getPolicyNetwork()(states).gather(dim=1, index=actions)
            nextQValue = self.getTargetNetwork()(nextStates).max(dim=1)[0].detach()

            # Set next q value to zero for end states
            doneMask = torch.ByteTensor(isDones).bool().to(self.device)
            nextQValue[doneMask] = 0.0

            targetQValue = rewards + self.gamma * nextQValue

            # loss = functional.mse_loss(currentQValue, targetQValue.unsqueeze(1))
            criterion = nn.SmoothL1Loss()
            loss = criterion(currentQValue, targetQValue.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.getPolicyNetwork().parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

    def getTargetNetwork(self):
        return NotImplementedError()

    def getPolicyNetwork(self):
        return NotImplementedError()

    def getEnvironment(self):
        raise NotImplementedError()

    def getAgent(self):
        raise NotImplementedError()