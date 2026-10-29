"""The end-to-end alignment pipeline: build task, fit reference, run DPO, evaluate."""

from __future__ import annotations

from dataclasses import dataclass

from preference_optimization.domain import Domain, demonstrations, preferences
from preference_optimization.dpo import DPOConfig, train_dpo
from preference_optimization.evaluation import AlignmentReport, evaluate
from preference_optimization.policy import SoftmaxPolicy
from preference_optimization.sft import SFTConfig, behavior_clone


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level knobs for the demonstration and preference data and the two stages."""

    demonstrations: int = 600
    preferences: int = 4000
    held_out_preferences: int = 800
    sft_temperature: float = 1.0
    beta: float = 0.3
    seed: int = 0


@dataclass(frozen=True)
class PipelineResult:
    """The trained policies and their reports."""

    domain: Domain
    reference: SoftmaxPolicy
    policy: SoftmaxPolicy
    report: AlignmentReport


def run_pipeline(config: PipelineConfig | None = None) -> PipelineResult:
    """Fit a reference by behavior cloning, align it with DPO, and evaluate."""

    config = config or PipelineConfig()
    domain = Domain.build(seed=config.seed)
    sft_data = demonstrations(
        domain, config.demonstrations, temperature=config.sft_temperature, seed=config.seed + 1
    )
    reference = behavior_clone(domain, sft_data, SFTConfig(seed=config.seed + 2))
    train_preferences = preferences(domain, config.preferences, seed=config.seed + 3)
    held_out = preferences(domain, config.held_out_preferences, seed=config.seed + 4)
    policy = train_dpo(
        domain, reference, train_preferences, DPOConfig(beta=config.beta, seed=config.seed + 5)
    )
    report = evaluate(domain, policy, reference, held_out, beta=config.beta)
    return PipelineResult(domain=domain, reference=reference, policy=policy, report=report)
