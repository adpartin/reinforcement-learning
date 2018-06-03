# Multi-armed Bandits

- Objective is to maximize the expected total reward over some time period, for example, over 1000 action selections, or time steps.
- If we call `q_a` the optical value for a specific action is the expected reward given than `a` is selected.
- A greedy action is an action that select the action with the highest `q_a` at each time step. How do we estimate those values ?

## action-value method

One simple way to do that is to estimate each action using the mean of the reward obatin for these action in the past experience.

```
q_a = Sum of reward obtain for a / Number a was chosen
```

If we maintain an estimate of `q_a` this way, we can just choose the action a with the hight value `q_a`.

To encourage a bit of exploration though, we are going to choose with a proba equal to (1-epsilon) and random action.

## A simple bandit algo.

We can easily update the action values estimate

```
q_a(t+1) = q_a(t)+ alpha(R-q_a)
```

where R is the reward obtain when chosing action a at time t. `alpha` is often called the learning rate. It can be set to `1/N_a` here.

This update rule can be generalize as followed.

```
NewEstimate -> OldEstimate + alpha(Target-OldEstimate)
```

Given this rule, the pseudo code is

```
q_a = 0
N_a = 1
while eternity:
   if prob > epsilon:
      a = argmax(q_a)
   else:
      a = random
    reward = env.step(a)
    q_a += 1/N_a(R-q_a)
    N_a +=1
```

## Optimistic Initial Values

The algo depends on our initial estimate for `q_a`. One way to encourage exploration is the beginning is too you a large initialization value, larger than the maximum expected reward.

## Upper-Confidence-Bound Action Selection

The idea of this upper confidence bound (UCB) action selection is that the square-root term is a measure of the uncertainty or variance in the estimate of a’s value. The quantity being max’ed over is thus a sort of upper bound on the possible true value of action a, with c determining the confidence level.



