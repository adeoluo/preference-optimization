<h1 align="center">Preference Optimization</h1>

<p align="center">
  <em>Direct Preference Optimization built from scratch: a supervised reference policy, a Bradley-Terry preference dataset, and an exact, hand-derived DPO gradient that aligns the policy to the hidden reward.</em>
</p>

<p align="center">
  <a href="https://github.com/adeoluo/preference-optimization/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/adeoluo/preference-optimization/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/adeoluo/preference-optimization/releases"><img alt="Release" src="https://img.shields.io/github/v/release/adeoluo/preference-optimization"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

---

## Motivation

Aligning a language model to human preferences is the central problem of modern
post-training. **Direct Preference Optimization** (Rafailov et al., 2023) replaces the
reinforcement-learning stage of RLHF with a single classification loss on preference pairs: no
reward model to train separately, no rollouts, no clipped policy gradient. It rests on the
insight that the optimal KL-regularized policy has a closed form, so the reward can be folded
into the policy itself and fit directly from comparisons. This project implements the whole
pipeline **from scratch** on a tractable alignment task: a supervised reference fit by
**behavior cloning**, a **Bradley-Terry** preference dataset, and the **exact DPO gradient**,
all with no automatic differentiation.

## Architecture

<p align="center">
  <img src="docs/architecture/architecture.svg" alt="Architecture diagram" width="860">
</p>

<sub>Legend: amber = entry points and external systems, indigo = core logic, green = data and state, blue = validation and control, grey = observability and outputs.</sub>

A hidden gold reward defines the truth. A low-temperature teacher produces demonstrations that
a reference policy is cloned from, and a Bradley-Terry annotator produces noisy preferences
that DPO optimizes. The learner never sees the reward. The pipeline runs once at startup
(600 demonstrations, 4000 training pairs, 800 held-out pairs); the evaluation compares the
aligned policy with the frozen reference against the gold reward, and the HTTP service serves
the report and validated per-prompt scores. See the [ADRs](docs/adr/) for the key
decisions.

### The method

DPO aligns a policy `p` toward a frozen reference `q` using preference pairs. For a chosen
response `w` and a rejected response `l` under prompt `x`, with regularization strength `beta`,
define the margin

```
m = beta * [ (log p(w|x) - log q(w|x)) - (log p(l|x) - log q(l|x)) ]
```

and minimize `-log sigmoid(m)`. The reward that DPO implicitly fits is `beta * log (p / q)`.
For the log-linear policy used here the expected-feature terms in the two log-probability
gradients cancel, so the loss gradient is exactly

```
dL/dtheta = -sigmoid(-m) * beta * (phi(x, w) - phi(x, l)),
```

which this project implements directly. The pipeline has three stages: behavior cloning fits
the reference by penalized maximum likelihood, the preference data is sampled from a
Bradley-Terry annotator, and DPO aligns a copy of the reference.

### Components

- **domain**: the alignment task, the hidden gold reward, the low-temperature teacher, and the
  Bradley-Terry preference annotator.
- **policy**: a log-linear softmax policy with exact distributions, log-probabilities,
  expected features, and sampling.
- **sft**: behavior cloning of the reference by penalized maximum likelihood.
- **dpo**: the DPO loss and the exact, hand-derived gradient, with a training loop.
- **evaluation**: reward, win-rate, preference accuracy, KL, and implicit reward correlation.
- **pipeline**: the end-to-end run from task to aligned policy.

## Results

> Reproduce with `python -m preference_optimization evaluate`. All metrics are computed against
> the hidden gold reward, which the learner never observes.

| Metric | Value |
|---|---:|
| **reward gap closed (reference to optimal)** | **86.1%** |
| expected reward (reference to aligned) | 0.589 to 1.067 (optimal 1.144) |
| held-out preference accuracy | 0.863 |
| win-rate versus reference | 0.645 |
| implicit reward versus true reward (Pearson) | 0.767 |
| mean KL to reference | 0.782 |

Starting from a partially aligned supervised reference, DPO **closes 86 percent of the reward
gap to the optimal policy** while staying close to the reference in KL. The reward that DPO
never saw is recovered implicitly: the quantity `beta * log (p / q)` correlates with the true
reward at **r = 0.767**. A sweep over `beta` traces the reward versus divergence frontier
(smaller `beta` aligns harder, see the [evaluation report](docs/EVALUATION_REPORT.md)).

### Scale

`benchmarks/scale.py` aligns a fixed supervised reference with DPO on progressively larger
synthetic preference datasets (a reduced 15-epoch schedule for the sweep) and reports wall-clock
time, throughput, peak traced memory (`tracemalloc`), and held-out preference accuracy. Peak
memory grows linearly with the resident dataset while per-pair throughput stays roughly flat.

| preference pairs | seconds | pairs/sec | peak MiB | held-out accuracy |
|---:|---:|---:|---:|---:|
| 2,000 | 2.81 | 710 | 0.22 | 0.831 |
| 8,000 | 13.46 | 590 | 0.86 | 0.805 |
| 16,000 | 24.12 | 660 | 1.72 | 0.811 |

Accuracy here sits below the headline 0.863 because the sweep uses 15 epochs rather than the
full 200, trading a little alignment quality for a bounded runtime. Caveat: the data is
synthetic, this is a from-scratch Python implementation, and it is **not** benchmarked
against compiled numerical libraries; wall-clock and throughput were measured on a shared
development machine and will vary. Reproduce with `uv run python benchmarks/scale.py`.

## Quickstart

```bash
git clone https://github.com/adeoluo/preference-optimization.git && cd preference-optimization
make setup                    # install dev tooling + pre-commit hooks
make all                      # lint + type-check + test

python -m preference_optimization evaluate        # full pipeline and alignment report
python -m preference_optimization compare         # greedy choice before and after DPO
```

## Usage

### Demonstration

```text
$ python -m preference_optimization compare
Per-prompt greedy choice (reference -> aligned, with the gold-best response):
  prompt 0: reference r3 -> aligned r1  (gold-best r1, match)
  prompt 1: reference r0 -> aligned r4  (gold-best r4, match)
  ...
```

DPO moves the greedy choice onto the reward-optimal response for prompts where the reference
was wrong, using only noisy pairwise comparisons.

### Library

```python
from preference_optimization import run_pipeline, PipelineConfig

result = run_pipeline(PipelineConfig(beta=0.3))
print(result.report.format())
```

### HTTP service

A standard-library HTTP service runs the pipeline once at startup and serves alignment queries:

```bash
python -m preference_optimization serve --port 8000

curl localhost:8000/health
curl localhost:8000/report
curl -s -X POST localhost:8000/score -d '{"prompt": 0}'
```

See the [operations guide](docs/OPERATIONS.md) for the service contract and constraints. The
service also exposes `GET /ready` (readiness) and `GET /metrics` (per-route request, error, and
latency metrics in Prometheus text format). A production-style stress benchmark
(`uv run python benchmarks/stress.py`) reports throughput and p50/p95/p99 latency; the scaling
path and tradeoffs are in [docs/SCALING.md](docs/SCALING.md) and a full walk-through is in the
[case study](docs/CASE_STUDY.md).

## Design decisions

- **Direct Preference Optimization, not a reward model plus PPO.** A single closed-form loss on
  comparisons, with no separate reward model and no rollouts.
  ([ADR-0002](docs/adr/0002-direct-preference-optimization.md))
- **A log-linear policy with an exact gradient.** The expected-feature terms cancel in the DPO
  gradient, giving a clean, exact implementation that is easy to verify.
  ([ADR-0003](docs/adr/0003-log-linear-policy.md))
- **Recording decisions.** ([ADR-0001](docs/adr/0001-record-architecture-decisions.md))

## Testing

```bash
make test                     # uv run pytest
make coverage                 # pytest with an 80% coverage gate
```

The suite has 13 deterministic, offline tests in two modules under `tests/`:

- **Core** (`test_core.py`): policy distributions are normalized, a uniform policy is uniform,
  log-probabilities match the distribution, behavior cloning improves over uniform, DPO closes
  the reward gap and lowers the loss, the evaluation report is consistent, and degenerate
  inputs (zero win-rate trials, constant Pearson inputs) return zero.
- **HTTP service** (`test_service.py`): the handler is exercised without a socket for health
  and readiness, `/report`, `/score` (including a bad prompt), unknown routes, and metrics.

CI (`.github/workflows/ci.yml`) runs `ruff format --check`, `ruff check`, `mypy src`, and
`pytest` with an 80% coverage gate on Python 3.11 and 3.12, then builds the package and the
Docker image.

## Responsible AI

- **Model card:** [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- **Data card:** [docs/DATA_CARD.md](docs/DATA_CARD.md)
- **Threat model:** [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md)

## Limitations

- The task is a contextual bandit with a log-linear policy, chosen so every quantity is exact.
  A sequence-level language-model policy would need per-token log-probabilities and automatic
  differentiation.
- Preference data is simulated from a Bradley-Terry annotator over a known reward; real
  annotations carry different and often inconsistent noise.
- DPO can overfit to noisy preferences when `beta` is small and training is long; the reward
  versus divergence tradeoff is documented rather than hidden.

## Roadmap

- [ ] Identity Preference Optimization and a conservative DPO variant
- [ ] A token-level policy with teacher-forced log-probabilities
- [ ] Preference label noise ablations and calibration curves

## License

Released under the [MIT License](LICENSE).
