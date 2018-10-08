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


def compute_return(episode, gamma=1):
    Gs = []
    for i in range(len(episode)):
        tmp = episode.loc[i:]
        tmp.reset_index(inplace=True)
        Gs.append(
            sum([gamma**j * tmp.loc[j, 'reward'] for j in range(len(tmp))]))
    return Gs


def generate_episode(policy, env):
    step = []
    state = env.reset()
    while True:
        action = policy(state)
        nstate, reward, done, info = env.step(action)
        step.append([state, action, reward])
        if done:
            break
        state = nstate
    episode = pd.DataFrame(step, columns=['state', 'action', 'reward'])
    episode['return'] = compute_return(episode)
    return episode


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    data = []
    for e in tqdm(range(num_episodes)):
        df = generate_episode(policy, env)
        df['episode_id'] = e
        data.append(df)
    data = pd.concat(data)
    gps = data.groupby('state')
    Vs = gps['return'].sum() / gps.size()
    Vs = Vs.to_dict()
    return Vs


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(observation):
        pgreedy = epsilon / float(nA)
        if np.random.rand() < pgreedy:
            action = random.choice(range(nA))
        else:
            action = np.argmax(Q[observation])
        return action

    return policy_fn


def mc_control_epsilon_greedy(env,
                              num_episodes,
                              discount_factor=1.0,
                              epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    data = []
    Qs = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Qs, epsilon, env.action_space.n)
    for e in tqdm(range(num_episodes)):
        df = generate_episode(policy, env)
        df['episode_id'] = e
        data.append(df)

        # improve policy
        Q = pd.concat(data)
        Q.loc[:, 'a0'] = (Q.action == 0) * Q['return']
        Q.loc[:, 'a1'] = (Q.action == 1) * Q['return']
        Q = Q.groupby('state')['a0', 'a1'].mean()
        Qs.update({
            k: np.array([v['a0'], v['a1']])
            for k, v in Q.to_dict('index').items()
        })
        policy = make_epsilon_greedy_policy(Qs, epsilon, env.action_space.n)

    return Qs, policy


def _generate_episode(policy, env):
    step = []
    state = env.reset()
    while True:
        action = policy(state)
        nstate, reward, done, info = env.step(action)
        step.append([state, action, reward])
        if done:
            break
        state = nstate
    return step
