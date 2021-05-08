import torch

import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, title, xlabel, ylabel, movingAveragePeriod):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.values = []
        self.movingAveragePeriod = movingAveragePeriod

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
        self.plot()

    def _saveInterval(self):
        if len(self.values) < 100:
            return 10
        if len(self.values) < 500:
            return 50
        return 100

    def plot(self):
        if len(self.values) % self._saveInterval() == 0:
            plt.clf()

            plt.title(self.title)
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            plt.plot(self.values)

            movingAverage = self.calculateMovingAverage()
            plt.plot(movingAverage)
            plt.savefig(f'plots/ep_{len(self.values)}.png', dpi=300)

    def getMovingAverage(self):
        return self.calculateMovingAverage()[-1]
