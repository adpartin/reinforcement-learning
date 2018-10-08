import matplotlib
import sys
import pandas as pd
from collections import defaultdict
import os
ROOT_DIR = os.environ['ROOT_DIR']
sys.path.append(ROOT_DIR)
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random
from lib import plotting


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(state):
        p = np.random.rand()
        if p > epsilon:
            action = np.argmax(Q[state])
        else:
            action = random.choice(range(nA))
        return action

    return policy_fn


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    done = True
    episode = 0
    while True:
        if done:
            done = False
            t = 0
            state0 = env.reset()
        action0 = policy(state0)
        state1, reward, done, info = env.step(action0)
        if done:
            episode += 1
            if episode % 5 == 0:
                print('Finished episode {}'.format(episode))
        stats.episode_lengths[episode] = t
        stats.episode_rewards[episode] += reward
        action1 = policy(state1)
        tdtarget = reward + discount_factor * Q[state1][action1]
        Q[state0][action0] += alpha * (tdtarget - Q[state0][action0])
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        state0 = state1
        t += 1
        if episode == num_episodes - 1:
            break

    return Q, stats


def qlearning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    done = True
    episode = 0
    while True:
        if done:
            done = False
            t = 0
            state0 = env.reset()
        action0 = policy(state0)
        state1, reward, done, info = env.step(action0)
        if done:
            episode += 1
            if episode % 5 == 0:
                print('Finished episode {}'.format(episode))
        stats.episode_lengths[episode] = t
        stats.episode_rewards[episode] += reward
        # instead of using the policy, max out over Q v
        tdtarget = reward + discount_factor * np.max(Q[state1])
        Q[state0][action0] += alpha * (tdtarget - Q[state0][action0])
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        state0 = state1
        t += 1
        if episode == num_episodes - 1:
            break

    return Q, stats
