# Scaling - Preference Optimization

This note describes how the pipeline behaves as data and load grow, and the tradeoffs on the
path to a real alignment system.

## Where the time goes
- **Behavior cloning** is a penalized maximum-likelihood fit over the demonstrations: each
  update is one expected-feature computation over a prompt's responses. It finishes in a
  fraction of a second for hundreds of demonstrations.
- **DPO** is the dominant cost: each preference update computes four log-probabilities (chosen
  and rejected under the policy and the reference) and applies the exact gradient. Cost scales
  linearly with the number of preferences and epochs.
- **Serving** is read-only: a score query computes two distributions and the implicit reward
  over a fixed, small response set, which is sub-millisecond.

## Request-time scaling
The service is stateless and the policies are immutable after startup, so there is no shared
mutable state and no lock on the hot path. A stress benchmark
(`uv run python benchmarks/stress.py`) drives the score handler under concurrency and reports
over 13,000 requests per second with sub-millisecond median latency. Horizontal scaling is
adding replicas behind a load balancer.

## Data scaling
Training time is linear in the number of preferences and the number of epochs. Memory is the
single weight vector plus the task tensors, independent of the number of comparisons. The two
levers are the amount of preference data and beta.

## Tradeoffs on the path to a real system
- **Beta.** Smaller beta aligns harder and recovers the reward more faithfully but moves the
  policy further from the reference and is more prone to overfitting noisy preferences. The
  sweep in the evaluation report traces this frontier.
- **Bandit versus sequence model.** The log-linear contextual bandit keeps everything exact. A
  real language-model policy needs per-token log-probabilities and automatic differentiation,
  which trades transparency for scale.
- **Reward model versus DPO.** DPO removes the separate reward model and the reinforcement
  stage, at the cost of committing to the Bradley-Terry link and losing an explicit reward to
  inspect; the implicit reward recovers much of that visibility.

## Limits
Training is pure Python and single-threaded, ideal for study and modest data but not for
web-scale preference corpora. The controlled task bounds the conclusions; scaling to real
alignment is a data and modeling problem, not a throughput one.
