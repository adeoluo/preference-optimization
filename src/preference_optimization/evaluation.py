"""Evaluation metrics for preference optimization.

All metrics are computed against the hidden gold reward, which the learner never
sees. They quantify how much closer preference optimization moves the policy to the
reward-optimal behavior, how tightly it stays near the reference, and whether the
implicit reward that Direct Preference Optimization fits actually tracks the truth.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from preference_optimization.domain import Domain, Preference
from preference_optimization.policy import SoftmaxPolicy


@dataclass(frozen=True)
class AlignmentReport:
    """A summary of a policy relative to the reference and the gold reward."""

    expected_reward: float
    optimal_reward: float
    reference_reward: float
    win_rate_vs_reference: float
    preference_accuracy: float
    mean_kl_to_reference: float
    implicit_reward_correlation: float

    @property
    def reward_gap_closed(self) -> float:
        span = self.optimal_reward - self.reference_reward
        if span <= 0.0:
            return 0.0
        return (self.expected_reward - self.reference_reward) / span

    def format(self) -> str:
        return (
            f"expected reward {self.expected_reward:.4f} "
            f"(reference {self.reference_reward:.4f}, optimal {self.optimal_reward:.4f}; "
            f"gap closed {self.reward_gap_closed:.1%})  "
            f"win-rate vs reference {self.win_rate_vs_reference:.3f}  "
            f"preference accuracy {self.preference_accuracy:.3f}  "
            f"mean KL {self.mean_kl_to_reference:.3f}  "
            f"implicit/true reward r {self.implicit_reward_correlation:.3f}"
        )


def expected_reward(domain: Domain, policy: SoftmaxPolicy) -> float:
    """Average over prompts of the reward expected under the policy distribution."""

    total = 0.0
    for prompt in range(domain.n_prompts):
        distribution = policy.distribution(prompt)
        total += sum(
            probability * domain.true_reward(prompt, response)
            for response, probability in enumerate(distribution)
        )
    return total / domain.n_prompts


def optimal_reward(domain: Domain) -> float:
    return (
        sum(
            domain.true_reward(prompt, domain.best_response(prompt))
            for prompt in range(domain.n_prompts)
        )
        / domain.n_prompts
    )


def win_rate(
    domain: Domain, policy: SoftmaxPolicy, reference: SoftmaxPolicy, *, trials: int, seed: int
) -> float:
    """Fraction of sampled prompts where the policy sample beats the reference sample."""

    rng = random.Random(seed)
    wins = 0.0
    for _ in range(trials):
        prompt = rng.randrange(domain.n_prompts)
        policy_reward = domain.true_reward(prompt, policy.sample(prompt, rng))
        reference_reward = domain.true_reward(prompt, reference.sample(prompt, rng))
        if policy_reward > reference_reward:
            wins += 1.0
        elif policy_reward == reference_reward:
            wins += 0.5
    return wins / trials


def preference_accuracy(domain: Domain, policy: SoftmaxPolicy, data: list[Preference]) -> float:
    """Fraction of pairs where the policy ranks the truly better response higher."""

    if not data:
        return 0.0
    correct = 0
    for preference in data:
        better, worse = preference.chosen, preference.rejected
        if domain.true_reward(preference.prompt, worse) > domain.true_reward(
            preference.prompt, better
        ):
            better, worse = worse, better
        if policy.log_prob(preference.prompt, better) > policy.log_prob(preference.prompt, worse):
            correct += 1
    return correct / len(data)


def mean_kl(domain: Domain, policy: SoftmaxPolicy, reference: SoftmaxPolicy) -> float:
    """Mean over prompts of KL(policy || reference)."""

    total = 0.0
    for prompt in range(domain.n_prompts):
        policy_distribution = policy.distribution(prompt)
        reference_distribution = reference.distribution(prompt)
        divergence = 0.0
        for probability, reference_probability in zip(
            policy_distribution, reference_distribution, strict=True
        ):
            if probability > 0.0:
                divergence += probability * math.log(probability / reference_probability)
        total += divergence
    return total / domain.n_prompts


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    if variance_x == 0.0 or variance_y == 0.0:
        return 0.0
    return covariance / math.sqrt(variance_x * variance_y)


def implicit_reward_correlation(
    domain: Domain, policy: SoftmaxPolicy, reference: SoftmaxPolicy, *, beta: float
) -> float:
    """Pearson correlation of the DPO implicit reward with the hidden true reward."""

    implicit: list[float] = []
    true: list[float] = []
    for prompt in range(domain.n_prompts):
        for response in range(domain.n_responses):
            implicit.append(
                beta * (policy.log_prob(prompt, response) - reference.log_prob(prompt, response))
            )
            true.append(domain.true_reward(prompt, response))
    return _pearson(implicit, true)


def evaluate(
    domain: Domain,
    policy: SoftmaxPolicy,
    reference: SoftmaxPolicy,
    held_out: list[Preference],
    *,
    beta: float,
    trials: int = 4000,
    seed: int = 7,
) -> AlignmentReport:
    """Produce a full alignment report for a policy against its reference."""

    return AlignmentReport(
        expected_reward=expected_reward(domain, policy),
        optimal_reward=optimal_reward(domain),
        reference_reward=expected_reward(domain, reference),
        win_rate_vs_reference=win_rate(domain, policy, reference, trials=trials, seed=seed),
        preference_accuracy=preference_accuracy(domain, policy, held_out),
        mean_kl_to_reference=mean_kl(domain, policy, reference),
        implicit_reward_correlation=implicit_reward_correlation(
            domain, policy, reference, beta=beta
        ),
    )
