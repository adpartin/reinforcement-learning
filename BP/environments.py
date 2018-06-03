import numpy as np
import pandas as pd
import seaborn as sns


class RandomWalkBanditEnvironment(object):
    """
    Non stationary bandit env where the reward for each is given by a normal whose
    mean follow a random walk.
    """

    def __init__(self, nmachine=10):
        self.nmachine = 10
        self.setup_environment()

    def setup_environment(self):
        self.reward = np.array([0] * self.nmachine).astype('float')

    def step(self, action):
        self.reward += np.random.normal(0, 0.01, 10)
        return self.reward[action]


class BanditEnvironment(object):
    """
    Stationary bandit env.
    """

    def __init__(self, nmachine=10):
        self.nmachine = 10
        self.setup_environment()

    def setup_environment(self):
        self.means = np.random.normal(0, 1, self.nmachine)

    def step(self, action):
        reward = np.random.normal(self.means[action], 1)
        return reward

    def display(self):
        samples = [
            np.vstack([
                np.random.normal(self.means[action], 1, 1000), [action] * 1000
            ]) for action in range(self.nmachine)
        ]
        samples = np.hstack(samples).T
        samples = pd.DataFrame(samples, columns=['reward', 'action'])
        return sns.violinplot(x=samples.action, y=samples.reward)
