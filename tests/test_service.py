"""Tests for the HTTP service handler (exercised without a socket)."""

from __future__ import annotations

from preference_optimization.domain import Domain, demonstrations, preferences
from preference_optimization.dpo import DPOConfig, train_dpo
from preference_optimization.evaluation import evaluate
from preference_optimization.pipeline import PipelineResult
from preference_optimization.service import AlignmentService, handle
from preference_optimization.sft import SFTConfig, behavior_clone


def _service() -> AlignmentService:
    domain = Domain.build(seed=0)
    sft = demonstrations(domain, 400, temperature=1.0, seed=1)
    reference = behavior_clone(domain, sft, SFTConfig(seed=2))
    train = preferences(domain, 1500, seed=3)
    held_out = preferences(domain, 400, seed=4)
    policy = train_dpo(domain, reference, train, DPOConfig(beta=0.3, epochs=100, seed=5))
    report = evaluate(domain, policy, reference, held_out, beta=0.3, trials=1000)
    result = PipelineResult(domain=domain, reference=reference, policy=policy, report=report)
    return AlignmentService(result=result, beta=0.3)


def test_health_and_ready() -> None:
    service = _service()
    status, payload = handle("GET", "/health", {}, service)
    assert status == 200
    assert payload["prompts"] == service.result.domain.n_prompts
    status, payload = handle("GET", "/ready", {}, service)
    assert status == 200
    assert payload["status"] == "ready"


def test_report_endpoint() -> None:
    service = _service()
    status, payload = handle("GET", "/report", {}, service)
    assert status == 200
    assert payload["reference_reward"] <= payload["optimal_reward"]
    assert 0.0 <= payload["win_rate_vs_reference"] <= 1.0


def test_score_endpoint_returns_responses() -> None:
    service = _service()
    status, payload = handle("POST", "/score", {"prompt": 0}, service)
    assert status == 200
    assert len(payload["responses"]) == service.result.domain.n_responses
    probabilities = sum(item["policy_prob"] for item in payload["responses"])
    assert abs(probabilities - 1.0) < 1e-3


def test_score_rejects_bad_prompt() -> None:
    service = _service()
    for bad in (-1, 999, "x", True):
        status, payload = handle("POST", "/score", {"prompt": bad}, service)
        assert status == 400
        assert "error" in payload


def test_unknown_route_and_metrics() -> None:
    service = _service()
    status, _ = handle("GET", "/nope", {}, service)
    assert status == 404
    handle("GET", "/report", {}, service)
    rendered = service.metrics.render_prometheus()
    assert "service_requests_total" in rendered
    assert service.metrics.total_errors >= 1
