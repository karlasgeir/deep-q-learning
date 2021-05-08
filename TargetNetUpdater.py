class TargetNetUpdater:
    def __init__(self, targetNet, policyNet, updateFrequency):
        self.steps = 0
        self.updateFrequency = updateFrequency
        self.targetNet = targetNet
        self.policyNet = policyNet

    def updateTargetNetwork(self):
        self.targetNet.load_state_dict(self.policyNet.state_dict())

    def step(self):
        if self.steps % self.updateFrequency == 0:
            self.updateTargetNetwork()

        self.steps += 1
