import numpy as np
import torch
from itertools import count
from ExperienceReplay import ExperienceReplay
from EpsilonDecay import EpsilonDecay
from recommendations.RecommendationsDQNProblem import RecommendationsDQNProblem
from Plotter import Plotter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def runRecommendations():
    episodes = 30000
    lr = 0.001
    gamma = 0.99
    experienceReplayMemory = 10000
    learningExperienceStartThreshold = 5000
    batchSize = 32
    targetUpdateFrequency = 1000

    experienceReplay = ExperienceReplay(experienceReplayMemory, batchSize, learningExperienceStartThreshold)
    epsilonDecay = EpsilonDecay(1.0, 0.01, 0.000001)

    recommendationsProblem = RecommendationsDQNProblem(lr=lr, gamma=gamma, experienceReplay=experienceReplay, epsilonDecay=epsilonDecay, device=device, targetUpdateFrequency=targetUpdateFrequency)
    episodePlotter = Plotter('Training...', 'Episodes', 'Average Reward', 100)

    for episode in range(0, episodes):
        rewards = []
        stepPlotter = Plotter('Episode %d' % episode, 'Steps', 'Reward', 10000)
        recommendationsProblem.reset()

        for step in count():
            reward, done = recommendationsProblem.step()
            stepPlotter.append(reward)
            rewards.append(reward)

            if step % 10000 == 0:
                stepPlotter.plotMovingAverage(episode, step, True)

            if done:
                episodePlotter.append(np.average(rewards))
                episodePlotter.plot()
                break


if __name__ == '__main__':
    runRecommendations()

