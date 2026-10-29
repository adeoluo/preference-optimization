# Operations Guide - Preference Optimization

The service is a single stateless process that runs the alignment pipeline once at startup and
then answers read-only queries about the trained policies. It has zero runtime dependencies and
binds only the Python standard library HTTP server.

## Service contract

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/health` | none | `{"status": "ok", "prompts": int, "responses": int}` |
| GET | `/ready` | none | `{"status": "ready", "prompts": int}` once trained |
| GET | `/metrics` | none | Prometheus text: per-route requests, errors, latency histogram |
| GET | `/report` | none | the full alignment report |
| POST | `/score` | `{"prompt": int}` | per-response reward, probabilities, implicit reward |

## Startup
The pipeline (behavior cloning, then DPO) runs once when the process starts and takes roughly
fifteen seconds. Until it completes the process is not listening, so a readiness probe against
`/ready` fails closed. There is no warm-up traffic requirement.

## Constraints and limits
- `/score` requires an integer `prompt` within the task range; anything else is a 400.
- Request bodies that are not valid JSON return a 400 without touching the policies.
- The service is read-only: it never retrains in response to a request.

## Observability
- `/metrics` exposes `service_requests_total`, `service_errors_total`,
  `service_request_duration_ms_sum`, and a latency histogram, all labeled by route.
- Scrape `/metrics` with Prometheus and alert on error rate and p99 latency per route.

## Health and readiness
- **Liveness:** `/health` returns as soon as the process is up.
- **Readiness:** `/ready` returns ready only after the pipeline completes. Wire it to the
  orchestrator readiness probe so traffic is withheld until the policies are trained.

## Failure modes and responses
- **Malformed input:** rejected with a 400 and counted as a route error.
- **Unknown route:** 404, counted in metrics.
- **Process restart:** safe at any time; the pipeline is deterministic, so restarts reproduce
  identical policies and reports.

## Deployment
Run the container from the provided `Dockerfile` behind a reverse proxy that terminates TLS and
enforces connection and rate limits. The process is stateless, so scaling out is a matter of
running more replicas behind a load balancer (see [SCALING.md](SCALING.md)).
