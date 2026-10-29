"""A dependency-free HTTP service exposing the aligned policy.

The service runs the full alignment pipeline once at startup (fit a reference by
behavior cloning, then align it with Direct Preference Optimization) and then
answers read-only queries about the policies and the alignment report. A pure
handler is separated from the socket binding so it can be tested without a network.

Endpoints
---------
``GET  /health``    liveness and task shape.
``GET  /ready``     readiness once the pipeline has trained.
``GET  /metrics``   per-route request, error, and latency metrics (Prometheus text).
``GET  /report``    the full alignment report (rewards, win-rate, KL, correlation).
``POST /score``     body ``{"prompt": int}``; per-response rewards and probabilities.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from preference_optimization.observability import ServiceMetrics
from preference_optimization.pipeline import PipelineConfig, PipelineResult, run_pipeline

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class AlignmentService:
    """Holds the trained pipeline result and answers alignment queries."""

    result: PipelineResult
    beta: float
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    @classmethod
    def trained(cls, *, beta: float = 0.3, seed: int = 0) -> AlignmentService:
        result = run_pipeline(PipelineConfig(beta=beta, seed=seed))
        return cls(result=result, beta=beta)

    def report(self) -> JsonObject:
        report = self.result.report
        return {
            "expected_reward": round(report.expected_reward, 6),
            "reference_reward": round(report.reference_reward, 6),
            "optimal_reward": round(report.optimal_reward, 6),
            "reward_gap_closed": round(report.reward_gap_closed, 6),
            "win_rate_vs_reference": round(report.win_rate_vs_reference, 6),
            "preference_accuracy": round(report.preference_accuracy, 6),
            "mean_kl_to_reference": round(report.mean_kl_to_reference, 6),
            "implicit_reward_correlation": round(report.implicit_reward_correlation, 6),
        }

    def score(self, prompt: int) -> JsonObject:
        domain = self.result.domain
        reference_distribution = self.result.reference.distribution(prompt)
        policy_distribution = self.result.policy.distribution(prompt)
        responses = []
        for response in range(domain.n_responses):
            implicit = self.beta * (
                self.result.policy.log_prob(prompt, response)
                - self.result.reference.log_prob(prompt, response)
            )
            responses.append(
                {
                    "response": response,
                    "true_reward": round(domain.true_reward(prompt, response), 6),
                    "reference_prob": round(reference_distribution[response], 6),
                    "policy_prob": round(policy_distribution[response], 6),
                    "implicit_reward": round(implicit, 6),
                }
            )
        return {
            "prompt": prompt,
            "gold_best_response": domain.best_response(prompt),
            "policy_greedy_response": self.result.policy.greedy(prompt),
            "responses": responses,
        }


def _route(
    method: str, path: str, body: Mapping[str, Any], service: AlignmentService
) -> tuple[int, JsonObject]:
    domain = service.result.domain
    if method == "GET" and path == "/health":
        return 200, {"status": "ok", "prompts": domain.n_prompts, "responses": domain.n_responses}
    if method == "GET" and path == "/report":
        return 200, service.report()
    if method == "POST" and path == "/score":
        prompt = body.get("prompt")
        if not isinstance(prompt, int) or isinstance(prompt, bool):
            return 400, {"error": "'prompt' must be an integer"}
        if not 0 <= prompt < domain.n_prompts:
            return 400, {"error": f"'prompt' must be in [0, {domain.n_prompts})"}
        return 200, service.score(prompt)
    return 404, {"error": f"no route for {method} {path}"}


def handle(
    method: str, path: str, body: Mapping[str, Any], service: AlignmentService
) -> tuple[int, JsonObject]:
    """Route a request, recording per-route metrics, and return status and payload."""

    domain = service.result.domain
    if method == "GET" and path == "/ready":
        return 200, {"status": "ready", "prompts": domain.n_prompts}
    start = time.perf_counter()
    status, payload = _route(method, path, body, service)
    service.metrics.observe(
        f"{method} {path}", (time.perf_counter() - start) * 1000.0, error=status >= 400
    )
    return status, payload


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - exercised over sockets
    service: AlignmentService

    def _respond(self, status: int, payload: JsonObject) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = self.service.metrics.render_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status, payload = handle("GET", self.path, {}, self.service)
        self._respond(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "request body is not valid JSON"})
            return
        status, payload = handle("POST", self.path, body, self.service)
        self._respond(status, payload)

    def log_message(self, *_: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Start the HTTP service (blocking)."""

    handler = type("_BoundHandler", (_Handler,), {"service": AlignmentService.trained()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"preference-optimization service listening on http://{host}:{port}")
    server.serve_forever()
