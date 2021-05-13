from Agent import Agent
from DQNProblem import DQNProblem
from recommendations.RecommendationsDQN import RecommendationsDQN
from recommendations.RecommendationsEnvironment import RecommendationsEnvironment


class RecommendationsDQNProblem(DQNProblem):
    def __init__(self, lr, gamma, experienceReplay, epsilonDecay, device, targetUpdateFrequency):
        super().__init__(lr, gamma, experienceReplay, epsilonDecay, device, targetUpdateFrequency)

        self.environment = RecommendationsEnvironment(self.device)
        actionSpaceSize = 5

        self.policyNet = RecommendationsDQN(self.environment.getInputSize(), actionSpaceSize).to(self.device)
        self.targetNet = RecommendationsDQN(self.environment.getInputSize(), actionSpaceSize).to(self.device)
        self._initNeuralNets()

        self.agent = Agent(self.policyNet, self.epsilonDecay, actionSpaceSize, self.device)

    def getTargetNetwork(self):
        return self.targetNet

    def getPolicyNetwork(self):
        return self.policyNet

    def getEnvironment(self):
        return self.environment

    def getAgent(self):
        return self.agent