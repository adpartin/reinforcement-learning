import matplotlib
import sys

from collections import defaultdict
import os
ROOT_DIR = os.environ['ROOT_DIR']
sys.path.append(ROOT_DIR)
from collections import defaultdict
from tqdm import tqdm


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

    # Collect all episodes
    G = defaultdict(float)
    N = defaultdict(float)
    for i in tqdm(range(num_episodes)):
        episode = []
        state = env.reset()
        while True:
            action = policy(state)
            nstate, reward, done, _ = env.step(action)
            episode.append(nstate)
            if done:
                episode.append(reward)
                break
            state = nstate

        # update the return for each state visitedx
        for i, state in enumerate(episode[:-1][::-1]):
            G[state] += discount_factor**i * episode[-1]
            N[state] += 1

    V = defaultdict(float)
    for state, value in G.items():
        V[state] = 0 if N[state] == 0 else G[state] / N[state]

    return V
