class EpsilonDecay:
    currentStep = 0

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def epsilon(self):
        return max(self.end, self.start - self.currentStep * self.decay)

    def step(self):
        self.currentStep += 1
        return self.epsilon()
