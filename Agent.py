from Episode import Episode
import numpy as np


class Agent:
    def __init__(self, qValues, learningRate):
        self.qValues = qValues
        self.learningRate = learningRate

    # Completes an episode and udpate self.qValues
    # Assumes dim(A) == 2
    def run(self, episode: Episode):
        actionsTaken = []
        r = 0
        terminal = False
        while not terminal:
            state = episode.remainingAttempts - 1
            action = 0 if self.qValues[state,
                                       0] > self.qValues[state, 1] else 1
            actionsTaken.append(action)
            r, terminal = episode.step(action)

        for i in range(len(actionsTaken) - 1, -1, -1):
            a = actionsTaken[i]
            s = episode.totalAttempts - i - 1
            self.qValues[s][a] += self.learningRate * (r - self.qValues[s][a])

    # Uses np.argmax for finding action and vectorizes qValue updates
    # Less performant in simple cases
    def runWithNumpy(self, episode: Episode):
        actionsTaken = []
        r = 0
        terminal = False
        while not terminal:
            state = episode.remainingAttempts - 1
            action = np.argmax(self.qValues[state, :])
            actionsTaken.append(action)
            r, terminal = episode.step(action)

        s = [atp for atp in range(
            episode.remainingAttempts, episode.totalAttempts)]
        actionsTaken.reverse()
        updateIdx = (s, actionsTaken)
        self.qValues[updateIdx] += self.learningRate * \
            (r - self.qValues[updateIdx])
