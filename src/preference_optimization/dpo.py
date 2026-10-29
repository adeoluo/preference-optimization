"""Direct Preference Optimization with an exact, hand-derived gradient.

DPO replaces the reinforcement-learning stage of RLHF with a single classification
loss on preference pairs. For a chosen response ``w`` and a rejected response ``l``
under prompt ``x``, with policy ``p`` and frozen reference ``q``, define the margin

    m = beta * [ (log p(w|x) - log q(w|x)) - (log p(l|x) - log q(l|x)) ]

and minimize ``-log sigmoid(m)``. The implicit reward that DPO fits is
``beta * log (p / q)``. For a log-linear policy the expected-feature terms in the
two log-probability gradients cancel, so the gradient of the loss is simply

    dL/dtheta = -sigmoid(-m) * beta * (phi(x, w) - phi(x, l)),

a clean, exact expression that this module implements directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from preference_optimization.domain import Domain, Preference
from preference_optimization.policy import SoftmaxPolicy


@dataclass(frozen=True)
class DPOConfig:
    """Direct Preference Optimization hyperparameters."""

    epochs: int = 200
    learning_rate: float = 0.3
    beta: float = 1.0
    seed: int = 0


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _margin(
    policy: SoftmaxPolicy, reference: SoftmaxPolicy, preference: Preference, beta: float
) -> float:
    chosen = policy.log_prob(preference.prompt, preference.chosen) - reference.log_prob(
        preference.prompt, preference.chosen
    )
    rejected = policy.log_prob(preference.prompt, preference.rejected) - reference.log_prob(
        preference.prompt, preference.rejected
    )
    return beta * (chosen - rejected)


def preference_loss(
    policy: SoftmaxPolicy, reference: SoftmaxPolicy, data: list[Preference], beta: float
) -> float:
    """Mean DPO loss over a preference dataset."""

    if not data:
        return 0.0
    total = 0.0
    for preference in data:
        margin = _margin(policy, reference, preference, beta)
        total += -math.log(_sigmoid(margin)) if margin > -700 else -margin
    return total / len(data)


def train_dpo(
    domain: Domain, reference: SoftmaxPolicy, data: list[Preference], config: DPOConfig
) -> SoftmaxPolicy:
    """Align a copy of the reference policy to preferences with the exact DPO gradient."""

    import random

    policy = reference.copy()
    rng = random.Random(config.seed)
    order = list(data)
    for _ in range(config.epochs):
        rng.shuffle(order)
        for preference in order:
            margin = _margin(policy, reference, preference, config.beta)
            scale = _sigmoid(-margin) * config.beta
            phi_w = domain.feature(preference.prompt, preference.chosen)
            phi_l = domain.feature(preference.prompt, preference.rejected)
            for index in range(domain.dim):
                policy.weight[index] += config.learning_rate * scale * (phi_w[index] - phi_l[index])
    return policy
