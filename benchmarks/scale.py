"""A large-workload scale benchmark for Direct Preference Optimization.

Aligns a fixed supervised reference policy with DPO on progressively larger
synthetic preference datasets and reports, per workload, the wall-clock time,
throughput in preference pairs per second, the peak traced memory, and the
held-out preference accuracy. It complements ``benchmark.py`` (which profiles one
fixed-size run) by showing how the core DPO update scales as preference data grows.

Timing and memory are captured in separate phases on purpose. ``tracemalloc``
instruments every allocation and dominates the wall-clock of these tight numeric
loops, so it would distort the throughput numbers; the peak footprint is governed
by the resident preference dataset and the policy, not by the epoch count, so a
short traced pass measures it faithfully. Deterministic and offline; run with:

    uv run python benchmarks/scale.py
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

from preference_optimization.domain import Domain, Preference, demonstrations, preferences
from preference_optimization.dpo import DPOConfig, train_dpo
from preference_optimization.evaluation import preference_accuracy
from preference_optimization.policy import SoftmaxPolicy
from preference_optimization.sft import SFTConfig, behavior_clone

WORKLOADS: tuple[int, ...] = (2000, 8000, 16000)
_EPOCHS = 12
_MEM_EPOCHS = 1
_BETA = 0.3
_HELD_OUT = 800
_PREF_SEED = 3
_HELD_OUT_SEED = 4
_DPO_SEED = 5


@dataclass(frozen=True)
class ScalePoint:
    """One row of the scale table for a single workload size."""

    size: int
    seconds: float
    throughput: float
    peak_mib: float
    accuracy: float


def build_reference() -> tuple[Domain, SoftmaxPolicy, list[Preference]]:
    """Build the domain, a supervised reference policy, and a held-out split once."""

    domain = Domain.build(seed=0)
    reference = behavior_clone(
        domain, demonstrations(domain, 600, temperature=1.0, seed=1), SFTConfig(seed=2)
    )
    held_out = preferences(domain, _HELD_OUT, seed=_HELD_OUT_SEED)
    return domain, reference, held_out


def measure(
    domain: Domain,
    reference: SoftmaxPolicy,
    held_out: list[Preference],
    size: int,
    *,
    epochs: int = _EPOCHS,
    mem_epochs: int = _MEM_EPOCHS,
) -> ScalePoint:
    """Align the reference on ``size`` preference pairs and measure the cost."""

    tracemalloc.start()
    tracemalloc.reset_peak()
    data = preferences(domain, size, seed=_PREF_SEED)
    train_dpo(domain, reference, data, DPOConfig(epochs=mem_epochs, beta=_BETA, seed=_DPO_SEED))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    start = time.perf_counter()
    policy = train_dpo(
        domain, reference, data, DPOConfig(epochs=epochs, beta=_BETA, seed=_DPO_SEED)
    )
    seconds = time.perf_counter() - start

    return ScalePoint(
        size=size,
        seconds=seconds,
        throughput=size / seconds if seconds > 0 else 0.0,
        peak_mib=peak / 1048576,
        accuracy=preference_accuracy(domain, policy, held_out),
    )


def main() -> None:
    print("Synthetic preference data; a pure-Python, from-scratch DPO implementation.")
    print("Wall-clock is not competitive with compiled numerical libraries.")
    print(f"Running DPO for {_EPOCHS} epochs over each preference dataset.\n")

    domain, reference, held_out = build_reference()

    print(f"{'pairs':>8}  {'seconds':>8}  {'pairs/sec':>10}  {'peak MiB':>9}  {'accuracy':>8}")
    for size in WORKLOADS:
        point = measure(domain, reference, held_out, size)
        print(
            f"{point.size:>8}  {point.seconds:>8.3f}  {point.throughput:>10.0f}  "
            f"{point.peak_mib:>9.2f}  {point.accuracy:>8.3f}"
        )


if __name__ == "__main__":
    main()
