import copy

import numpy as np
import pandas as pd
import seaborn as sns


class GreedyBanditAgent(object):
    def __init__(self, epsilon=0.1, learning_rate=0.5, action_space=10, q0=0):
        self.action_space = action_space
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.action_values = np.array([q0] * self.action_space).astype('float')
        self.naction_values = np.array([1] * self.action_space).astype('float')
        self.rewards = []

    def choose(self):
        if np.random.rand() > self.epsilon:
            arr = np.argwhere(
                self.action_values == np.max(self.action_values)).flatten()
            action = np.random.choice(arr)
        else:
            action = np.random.choice(range(self.action_space))
        return action

    def update_agent(self, action, reward):
        self.rewards.append([action, reward])
        lr = self.learning_rate
        if lr is None:
            lr = 1. / float(self.naction_values[action])
        self.action_values[action] += lr * (
            reward - self.action_values[action])
        self.naction_values[action] += 1


class UCBBanditAgent(object):
    def __init__(self, c=0, learning_rate=0.5, action_space=10, q0=0):
        self.action_space = action_space
        self.c = c
        self.learning_rate = learning_rate
        self.action_values = np.array([q0] * self.action_space).astype('float')
        self.naction_values = np.array([1] * self.action_space).astype('float')
        self.rewards = []
        self.t = 1

    def choose(self):
        action_values = copy.deepcopy(self.action_values)
        action_values += self.c * np.sqrt(np.log(self.t) / self.naction_values)
        arr = np.argwhere(action_values == np.max(action_values)).flatten()
        action = np.random.choice(arr)
        return action

    def update_agent(self, action, reward):
        self.rewards.append([action, reward])
        lr = self.learning_rate
        if lr is None:
            lr = 1. / float(self.naction_values[action])
        self.action_values[action] += lr * (
            reward - self.action_values[action])
        self.naction_values[action] += 1
        self.t += 1
