import torch
import random


class ExperienceReplay:
    def __init__(self, limit, batchSize, startThreshold):
        self.limit = limit
        self.batchSize = batchSize
        self.startThreshold = startThreshold
        self.experiences = []

    def addExperience(self, experience):
        if len(self.experiences) >= self.limit:
            self.experiences.pop(0)
        self.experiences.append(experience)

    def getExperiences(self):
        if len(self.experiences) >= self.startThreshold:
            states, actions, rewards, nextStates, isDones = zip(*random.sample(self.experiences, self.batchSize))

            return torch.cat(states), torch.cat(actions), torch.cat(rewards), torch.cat(nextStates), isDones
        return None
