# Model / System Card - Preference Optimization

## Overview
- **What it is:** a from-scratch Direct Preference Optimization pipeline (behavior-cloned
  reference, Bradley-Terry preferences, exact DPO gradient) over a log-linear policy, with zero
  runtime dependencies.
- **Base model(s):** from-scratch; no pretrained components.
- **Version and date:** 1.0, 2026-01-01.

## Intended use
- **Primary use cases:** teaching and demonstrating preference optimization, the DPO objective
  and its implicit reward, and the reward versus divergence tradeoff; a reference implementation
  for experiments on beta and preference noise.
- **Primary users:** engineers and researchers studying post-training and alignment.

## Out-of-scope / prohibited use
- Aligning a real language model. The policy is a log-linear contextual bandit, not a sequence
  model, so results transfer in spirit rather than scale.

## Performance
- From a partially aligned reference, DPO closes **86.1%** of the reward gap to optimal,
  reaches **0.863** held-out preference accuracy, and the implicit reward correlates with the
  hidden true reward at **r = 0.767**, with mean KL to the reference of **0.782**. See
  [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).

## Risks and limitations
- DPO can overfit to noisy preferences when beta is small and training is long.
- The Bradley-Terry annotator is more consistent than real human labeling.
- The reward is linear in the features, which flatters a log-linear policy.

## Safety mitigations
- The HTTP service is read-only and validates that the prompt index is an integer in range
  before any computation runs.

## Environmental / cost footprint
- The full pipeline trains in roughly fifteen seconds of single-core CPU time with no
  accelerator, so the energy footprint is negligible.
