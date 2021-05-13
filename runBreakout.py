import torch
from itertools import count
from ExperienceReplay import ExperienceReplay
from EpsilonDecay import EpsilonDecay
from breakout.BreakoutDQNProblem import BreakoutDQNProblem
from Plotter import Plotter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def runBreakout():
    episodes = 30000
    lr = 0.0000625
    gamma = 0.99
    experienceReplayMemory = 50000
    learningExperienceStartThreshold = 10000
    batchSize = 32
    targetUpdateFrequency = 10000

    experienceReplay = ExperienceReplay(experienceReplayMemory, batchSize, learningExperienceStartThreshold)
    epsilonDecay = EpsilonDecay(1.0, 0.01, 0.000001)

    breakoutProblem = BreakoutDQNProblem(lr=lr, gamma=gamma, experienceReplay=experienceReplay,
                                         epsilonDecay=epsilonDecay, device=device,
                                         targetUpdateFrequency=targetUpdateFrequency)

    plotter = Plotter('Training...', 'Episodes', 'Reward', 100)

    totalSteps = 0
    for episode in range(0, episodes):
        breakoutProblem.reset()

        episodeReturn = 0
        for step in count():
            reward, done = breakoutProblem.step()
            episodeReturn += reward

            if done:
                totalSteps += step
                plotter.append(episodeReturn)
                movingAverage = plotter.getMovingAverage()
                print(f'Episode {episode + 1}:\n'
                      f'\tMoving avg: {movingAverage}\n'
                      f'\tReward: {episodeReturn}\n'
                      f'\tEpsilon: {epsilonDecay.epsilon()}\n'
                      f'\tEpisode steps: {step}\n'
                      f'\tTotal steps: {totalSteps}\n'
                  )

                break


if __name__ == '__main__':
    runBreakout()