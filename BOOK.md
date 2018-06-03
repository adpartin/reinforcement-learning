# BOOK

## Introduction

- The learner is not told which actions to take, as in most forms of machine learning, but instead must discover which actions yield the most reward by trying them.

- Four brick of einforcement learning system: a policy, a reward function, a value function, and, optionally, a model of the environment.

- A policy defines the learning agent’s way of behaving at a given time. Roughly speaking, a policy is a mapping from perceived states of the environ- ment to actions to be taken when in those states.

- A reward function maps each perceived state (state-action pair) to a specific value. A reinforcement learning agent’s sole objective is to maximize the total reward it receives in the long run.

- A value function specifies what is good in the long run. Quantify how much reward you can expect in average fron this specific state. Whereas a reward function indicates what is good in an immediate sense, a value function specifies what is good in the long run. 

- Given a state and action, the model might predict the resultant next state and next reward. Model-free reinforcement learning just use trial-and-error learning.



## Tabular solution method

### Multi-armed Bandits

- RL evaluates actions by experience instead of instructing what the right action to take is.

- Evaluative feedback depends entirely on the action taken (RL), whereas instructive feedback is independent of the action taken (Supervised learning).
