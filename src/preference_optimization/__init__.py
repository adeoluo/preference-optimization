"""Preference Optimization - Direct Preference Optimization from scratch."""

from preference_optimization.dpo import DPOConfig, preference_loss, train_dpo
from preference_optimization.evaluation import AlignmentReport, evaluate
from preference_optimization.pipeline import PipelineConfig, PipelineResult, run_pipeline
from preference_optimization.policy import SoftmaxPolicy
from preference_optimization.sft import SFTConfig, behavior_clone

__version__ = "0.1.0"

__all__ = [
    "AlignmentReport",
    "DPOConfig",
    "PipelineConfig",
    "PipelineResult",
    "SFTConfig",
    "SoftmaxPolicy",
    "behavior_clone",
    "evaluate",
    "preference_loss",
    "run_pipeline",
    "train_dpo",
]
