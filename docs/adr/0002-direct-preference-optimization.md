# ADR-0002: Direct Preference Optimization instead of a reward model and PPO

- **Status:** Accepted
- **Date:** 2026-01-01

## Context
The standard RLHF recipe has three stages: supervised fine-tuning, training a reward model on
human preference comparisons, and optimizing the policy against that reward model with a
clipped policy gradient such as PPO while a KL penalty keeps the policy near the reference.
That pipeline is powerful but heavy: it needs a separate reward model, on-policy rollouts, a
value function, and careful tuning of the reinforcement-learning stage. We want a transparent,
dependency-free implementation that still captures the essential alignment problem.

Direct Preference Optimization observes that the optimal KL-regularized policy has a closed
form, which can be rearranged so the reward is expressed through the policy and the reference.
Substituting that expression into the Bradley-Terry preference likelihood turns alignment into
a single classification loss on preference pairs, with no reward model and no rollouts.

## Decision
Implement **Direct Preference Optimization**. Fit a reference by behavior cloning, sample a
Bradley-Terry preference dataset, and minimize `-log sigmoid(margin)` where the margin is the
beta-scaled difference of the chosen and rejected log-probability ratios between the policy and
the frozen reference. The implicit reward is `beta * log (policy / reference)`.

## Consequences
- The whole alignment stage is one convex-looking classification loss over pairs, trainable by
  stochastic gradient descent with an exact gradient and no auxiliary models.
- The reward is never materialized, yet it is recovered implicitly; the evaluation measures the
  correlation of the implicit reward with the hidden true reward.
- DPO can overfit to noisy preferences when beta is small and training runs long. The reward
  versus divergence tradeoff is swept and documented rather than hidden, and beta is the lever.
