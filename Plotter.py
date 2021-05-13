import torch
import matplotlib.pyplot as plt
import os


class Plotter:
    def __init__(self, title, xlabel, ylabel, movingAveragePeriod, directory='plots'):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.values = []
        self.movingAveragePeriod = movingAveragePeriod
        self.directory = directory

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

    def calculateMovingAverage(self):
        values = torch.tensor(self.values, dtype=torch.float)
        if len(values) >= self.movingAveragePeriod:
            movingAverage = values.unfold(dimension=0, size=self.movingAveragePeriod, step=1) \
                .mean(dim=1).flatten(start_dim=0)
            movingAverage = torch.cat((torch.zeros(self.movingAveragePeriod - 1), movingAverage))
            return movingAverage.numpy()
        else:
            movingAverage = torch.zeros(len(values))
            return movingAverage.numpy()

    def append(self, value):
        self.values.append(value)

    def _saveInterval(self):
        if len(self.values) < 100:
            return 10
        if len(self.values) < 500:
            return 50
        return 100

    def plot(self, step=None):
        plt.clf()

        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.plot(self.values)

        filename = os.path.join(self.directory, f'ep_{len(self.values)}')

        if step is not None:
            filename = '%s_%d' % (filename, step)

        plt.savefig(f'{filename}.png', dpi=300)

    def plotMovingAverage(self, episode, step=None, force=False):
        if len(self.values) % self._saveInterval() == 0 or force:
            plt.clf()

            plt.title(self.title)
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            plt.plot(self.values)

            movingAverage = self.calculateMovingAverage()
            plt.plot(movingAverage)

            filename = os.path.join(self.directory, f'ep_{episode}')

            if step is not None:
                filename = '%s_%d' % (filename, step)

            plt.savefig(f'{filename}.png', dpi=300)

    def getMovingAverage(self):
        return self.calculateMovingAverage()[-1]
