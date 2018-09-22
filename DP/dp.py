import numpy as np


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        nextV = np.zeros(env.nS)
        for state in range(env.nS):
            newv = 0
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    newv += policy[state, action] * prob * (
                        reward + discount_factor * V[next_state])
            nextV[state] = newv
        if np.abs(np.sum(nextV - V)) < theta:
            break
        V = nextV
    return np.array(V)


def greedy_policy(env, V, discount_factor=1.0):
    Q = np.zeros([env.nS, env.nA])
    for state in range(env.nS):
        for action in range(env.nA):
            newQ = 0
            for prob, next_state, reward, done in env.P[state][action]:
                newQ += prob * (reward + discount_factor * V[next_state])
            Q[state, action] = newQ
    greedy_action = np.argmax(Q, axis=1)
    npolicy = np.zeros([env.nS, env.nA])
    for state in range(env.nS):
        npolicy[state, greedy_action[state]] = 1
    return npolicy


def policy_improvement(env, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI env.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        # Implement this!
        V = policy_eval(policy, env, discount_factor=discount_factor)
        # act gridy on that policy
        npolicy = greedy_policy(env, V)

        if np.mean(npolicy - policy == 0.0) == 1.0:
            break
        policy = npolicy
    return policy, V


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    Vstar = np.zeros(env.nS)
    while True:
        nextVstar = np.zeros(env.nS)
        for state in range(env.nS):
            newv = []
            for action in range(env.nA):
                tmp = 0
                for prob, next_state, reward, done in env.P[state][action]:
                    tmp += prob * (
                        reward + discount_factor * Vstar[next_state])
                newv.append(tmp)
            nextVstar[state] = max(newv)
        if np.abs(np.sum(nextVstar - Vstar)) < theta:
            break
        Vstar = nextVstar
    policy = greedy_policy(env, Vstar)
    return policy, np.array(Vstar)


def value_iteration_for_gamblers(p_h, theta=0.0001, discount_factor=1.0):
    """
    Args:
        p_h: Probability of the coin coming up heads
    """
    # two dummy states 0, 100 termination state
    nstate = 101
    rewards = np.zeros(nstate)
    rewards[100] = 1
    Vstar = np.zeros(nstate)

    while True:
        nextVstar = np.zeros(nstate)
        for state in range(1, 100):
            vstars = [0]
            for action in range(1, min(state, 100 - state) + 1):
                tmp = p_h * (rewards[state + action] + discount_factor *
                             Vstar[state + action]) + (1 - p_h) * (
                                 rewards[state - action] +
                                 discount_factor * Vstar[state - action])
                vstars.append(tmp)
            nextVstar[state] = max(vstars)
        if np.abs(np.sum(nextVstar - Vstar)) < theta:
            Vstar = nextVstar
            break
        Vstar = nextVstar

    policy = np.zeros(nstate)
    for state in range(1, 100):
        qsa = []
        actions = []
        for action in range(1, min(state, 100 - state) + 1):
            tmp = p_h * (rewards[state + action] + discount_factor *
                         Vstar[state + action]) + (1 - p_h) * (
                             rewards[state - action] +
                             discount_factor * Vstar[state - action])
            qsa.append(tmp)
            actions.append(action)
        policy[state] = actions[np.argmax(qsa)]
    return policy, Vstar
