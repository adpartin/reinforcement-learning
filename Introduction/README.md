## Introduction

### Learning Goals

- Understand the Reinforcement Learning problem and how it differs from Supervised Learning

### Summary

- Reinforcement Learning (RL) is concerned with goal-directed learning and decision-making.
- In RL an agent learns from experiences it gains by interacting with the environment. In Supervised Learning we cannot affect the environment.
- In RL rewards are often delayed in time and the agent tries to maximize a long-term goal. For example, one may need to make seemingly suboptimal moves to reach a winning position in a game.
- An agent interacts with the environment via states, actions and rewards.

### Note from the chapter 1

- The learner is not told which actions to take, as in most forms of machine learning, but instead must discover which actions yield the most reward by trying them.

- Four brick of einforcement learning system: a policy, a reward function, a value function, and, optionally, a model of the environment.

- A policy defines the learning agent’s way of behaving at a given time. Roughly speaking, a policy is a mapping from perceived states of the environ- ment to actions to be taken when in those states.

- A reward function maps each perceived state (state-action pair) to a specific value. A reinforcement learning agent’s sole objective is to maximize the total reward it receives in the long run.

- A value function specifies what is good in the long run. Quantify how much reward you can expect in average fron this specific state. Whereas a reward function indicates what is good in an immediate sense, a value function specifies what is good in the long run. 

- Given a state and action, the model might predict the resultant next state and next reward. Model-free reinforcement learning just use trial-and-error learning.

### Lectures & Readings

**Required:**

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/sutton/book/bookdraft2017june.pdf) - Chapter 1: The Reinforcement Learning Problem
- David Silver's RL Course Lecture 1 - Introduction to Reinforcement Learning ([video](https://www.youtube.com/watch?v=2pWv7GOvuf0), [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/intro_RL.pdf))
- [OpenAI Gym Tutorial](https://gym.openai.com/docs)

**Optional:**

N/A


### Exercises

- [Work through the OpenAI Gym Tutorial](https://gym.openai.com/docs)
