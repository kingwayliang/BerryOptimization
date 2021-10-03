import random


class Episode:

    # Assume same rewards and probs across state space
    # For the general case, each state can have its own set of actions with unique rewards and probs
    def __init__(self, totalAttempts: int, rewards: list[float], probabilities: list[float]):
        self.totalAttempts = totalAttempts
        self.remainingAttempts = totalAttempts
        self.rewards = rewards
        self.probabilities = probabilities

    # Given chosen action, returns reward and whether episode ends
    def step(self, action):
        self.remainingAttempts -= 1
        success = random.random() < self.probabilities[action]
        r = self.rewards[action] if success else 0
        return r, success or self.remainingAttempts == 0
