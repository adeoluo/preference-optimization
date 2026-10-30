"""Tests for the domain, policy, SFT, DPO, and evaluation."""

from __future__ import annotations

import math

from preference_optimization.domain import Domain, demonstrations, preferences
from preference_optimization.dpo import DPOConfig, preference_loss, train_dpo
from preference_optimization.evaluation import (
    _pearson,
    evaluate,
    expected_reward,
    optimal_reward,
    win_rate,
)
from preference_optimization.policy import SoftmaxPolicy
from preference_optimization.sft import SFTConfig, behavior_clone


def test_policy_distribution_is_normalized() -> None:
    domain = Domain.build(seed=1)
    policy = SoftmaxPolicy(domain=domain, weight=[0.3] * domain.dim)
    for prompt in range(domain.n_prompts):
        distribution = policy.distribution(prompt)
        assert abs(sum(distribution) - 1.0) < 1e-9
        assert all(probability >= 0.0 for probability in distribution)


def test_uniform_policy_has_uniform_distribution() -> None:
    domain = Domain.build(seed=2)
    policy = SoftmaxPolicy(domain=domain)
    distribution = policy.distribution(0)
    assert all(abs(p - 1.0 / domain.n_responses) < 1e-9 for p in distribution)


def test_log_prob_matches_distribution() -> None:
    domain = Domain.build(seed=3)
    policy = SoftmaxPolicy(domain=domain, weight=[0.5, -0.2] + [0.1] * (domain.dim - 2))
    distribution = policy.distribution(1)
    for response in range(domain.n_responses):
        assert abs(math.exp(policy.log_prob(1, response)) - distribution[response]) < 1e-9


def test_behavior_cloning_improves_over_uniform() -> None:
    domain = Domain.build(seed=4)
    data = demonstrations(domain, 400, temperature=1.0, seed=5)
    reference = behavior_clone(domain, data, SFTConfig(seed=6))
    uniform = SoftmaxPolicy(domain=domain)
    assert expected_reward(domain, reference) > expected_reward(domain, uniform)


def test_dpo_closes_reward_gap_and_lowers_loss() -> None:
    domain = Domain.build(seed=0)
    sft = demonstrations(domain, 600, temperature=1.0, seed=1)
    reference = behavior_clone(domain, sft, SFTConfig(seed=2))
    train = preferences(domain, 2000, seed=3)
    policy = train_dpo(domain, reference, train, DPOConfig(beta=0.3, epochs=120, seed=5))

    assert expected_reward(domain, policy) > expected_reward(domain, reference)
    assert expected_reward(domain, policy) <= optimal_reward(domain) + 1e-9
    assert preference_loss(policy, reference, train, 0.3) < preference_loss(
        reference, reference, train, 0.3
    )


def test_evaluate_report_is_consistent() -> None:
    domain = Domain.build(seed=0)
    sft = demonstrations(domain, 400, temperature=1.0, seed=1)
    reference = behavior_clone(domain, sft, SFTConfig(seed=2))
    train = preferences(domain, 1500, seed=3)
    held_out = preferences(domain, 400, seed=4)
    policy = train_dpo(domain, reference, train, DPOConfig(beta=0.3, epochs=120, seed=5))
    report = evaluate(domain, policy, reference, held_out, beta=0.3, trials=1000)

    assert 0.0 <= report.win_rate_vs_reference <= 1.0
    assert 0.0 <= report.preference_accuracy <= 1.0
    assert report.mean_kl_to_reference >= 0.0
    assert report.reference_reward <= report.optimal_reward


def test_win_rate_zero_trials_is_zero() -> None:
    # Regression: trials=0 divided by zero.
    domain = Domain.build(seed=0)
    policy = SoftmaxPolicy(domain=domain, weight=[0.1] * domain.dim)
    reference = SoftmaxPolicy(domain=domain, weight=[0.0] * domain.dim)
    assert win_rate(domain, policy, reference, trials=0, seed=0) == 0.0


def test_pearson_degenerate_inputs_are_zero() -> None:
    # Regression: empty or single-point inputs divided by zero.
    assert _pearson([], []) == 0.0
    assert _pearson([1.0], [2.0]) == 0.0
