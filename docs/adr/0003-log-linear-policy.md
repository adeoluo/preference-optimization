# ADR-0003: A log-linear policy with an exact gradient

- **Status:** Accepted
- **Date:** 2026-01-01

## Context
DPO is usually applied to a neural language-model policy whose per-token log-probabilities are
produced by automatic differentiation. We want the same objective but with a model simple
enough to keep the project dependency-free, exact, and fully testable, while still exercising
every part of the method.

## Decision
Use a **log-linear softmax policy** over the candidate responses for each prompt, with a single
weight vector shared across prompts and features that depend on both the prompt and the
response. The log-probability, the full distribution, the expected feature vector, and the DPO
gradient are then available in closed form. A convenient consequence is that the
expected-feature terms in the gradients of the chosen and rejected log-probabilities cancel, so
the DPO loss gradient reduces to `-sigmoid(-margin) * beta * (phi_chosen - phi_rejected)`.

## Consequences
- Training is a few dozen lines of pure Python with no tensor library and no autodiff, and the
  gradient can be checked by hand.
- The task is a contextual bandit rather than a sequence model, so results transfer in spirit
  rather than scale; this is recorded as a limitation and a roadmap item.
- Because the policy is log-linear, the reference fit by behavior cloning and the aligned policy
  share the same representation, which makes the KL and implicit-reward comparisons clean.
