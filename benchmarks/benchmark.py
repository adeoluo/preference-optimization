"""A deterministic micro-benchmark for the alignment pipeline.

Measures behavior-cloning and DPO training time, reports the alignment metrics, and
sweeps the DPO regularization strength beta to show the reward versus divergence
tradeoff. Deterministic and offline; run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

from preference_optimization.domain import Domain, demonstrations, preferences
from preference_optimization.dpo import DPOConfig, train_dpo
from preference_optimization.evaluation import evaluate
from preference_optimization.sft import SFTConfig, behavior_clone


def main() -> None:
    domain = Domain.build(seed=0)
    sft_data = demonstrations(domain, 600, temperature=1.0, seed=1)

    start = time.perf_counter()
    reference = behavior_clone(domain, sft_data, SFTConfig(seed=2))
    sft_ms = (time.perf_counter() - start) * 1000

    train = preferences(domain, 4000, seed=3)
    held_out = preferences(domain, 800, seed=4)

    start = time.perf_counter()
    policy = train_dpo(domain, reference, train, DPOConfig(beta=0.3, seed=5))
    dpo_ms = (time.perf_counter() - start) * 1000
    report = evaluate(domain, policy, reference, held_out, beta=0.3)

    print(f"Behavior cloning ({len(sft_data)} demos): {sft_ms:.0f} ms")
    print(f"DPO ({len(train)} preferences, 200 epochs): {dpo_ms:.0f} ms")
    print(f"  {report.format()}")
    print("beta sweep (reward versus divergence tradeoff):")
    for beta in (0.1, 0.2, 0.3, 0.5, 1.0):
        swept = train_dpo(domain, reference, train, DPOConfig(beta=beta, seed=5))
        swept_report = evaluate(domain, swept, reference, held_out, beta=beta)
        print(
            f"  beta={beta:<4} gap closed {swept_report.reward_gap_closed:>6.1%}  "
            f"mean KL {swept_report.mean_kl_to_reference:>5.2f}  "
            f"implicit/true r {swept_report.implicit_reward_correlation:>5.3f}"
        )


if __name__ == "__main__":
    main()
