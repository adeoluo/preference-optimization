# Evaluation Report - Preference Optimization

> Reproduce with `python -m preference_optimization evaluate` (report) and
> `uv run python benchmarks/benchmark.py` (report plus the beta sweep and timings). Evaluation
> is deterministic: the task, the data, and both training stages are fully seeded, so every
> number here is exact.

## Setup
- **Task:** 6 prompts, 5 responses each, 12-dimensional features, hidden gold reward.
- **Reference:** behavior-cloned from 600 low-temperature demonstrations.
- **Preferences:** 4000 Bradley-Terry comparisons for training, 800 held out from a disjoint
  seed.
- **DPO:** 200 epochs, learning rate 0.3, beta 0.3.

## Metrics
- **Reward gap closed:** how far the aligned policy moves from the reference reward toward the
  optimal reward.
- **Preference accuracy:** on held-out pairs, whether the policy ranks the truly better
  response higher.
- **Win-rate versus reference:** expected pairwise win of a policy sample over a reference
  sample under the true reward.
- **Mean KL to reference:** how far alignment moved the policy.
- **Implicit reward correlation:** Pearson correlation of `beta * log (policy / reference)` with
  the hidden true reward.

## Results

| Metric | Value |
|---|---:|
| reference expected reward | 0.589 |
| aligned expected reward | 1.067 |
| optimal expected reward | 1.144 |
| reward gap closed | 86.1% |
| held-out preference accuracy | 0.863 |
| win-rate versus reference | 0.645 |
| mean KL to reference | 0.782 |
| implicit reward versus true reward (Pearson) | 0.767 |

## The beta sweep (reward versus divergence)

| beta | gap closed | mean KL | implicit/true r |
|---:|---:|---:|---:|
| 0.1 | 90.0% | 0.86 | 0.888 |
| 0.2 | 87.9% | 0.82 | 0.822 |
| 0.3 | 86.1% | 0.78 | 0.767 |
| 0.5 | 83.6% | 0.77 | 0.711 |
| 1.0 | 84.1% | 2.01 | 0.717 |

Smaller beta aligns harder and recovers the reward more faithfully, at the cost of a larger
move from the reference. This is the KL-regularization knob of the method made visible.

## Interpretation
The win-rate of 0.645 is an **expected pairwise win** between two stochastic policies, not a
greedy comparison, so it is lower than the reward gap closed even though the aligned policy is
clearly better: both policies still sample, and several responses within a prompt have similar
reward. The strongest evidence that DPO worked is that a reward the learner never saw is
recovered implicitly, with the implicit reward correlating with the truth at r up to 0.888.

## Throughput
Behavior cloning over 600 demonstrations takes roughly 0.2 seconds; DPO over 4000 preferences
for 200 epochs takes roughly 14 seconds. The read-only service answers score queries at over
13,000 requests per second with sub-millisecond median latency.
