# Threat Model - Preference Optimization

## Assets
- Service availability, the integrity of the trained policies, and predictable resource use.

## Trust boundaries
- The HTTP service accepts untrusted JSON from clients. The task, demonstrations, and
  preferences are trusted and generated in-process; there is no external data ingestion at
  request time.

## Threats and mitigations
- **Malformed requests.** The `/score` endpoint requires an integer `prompt` within the task
  range; non-integers, booleans, and out-of-range values are rejected with a 400 before any
  work runs. Bodies that are not valid JSON return a 400.
- **Resource exhaustion.** Every request does a constant amount of work over a fixed, small
  number of responses, so there is no input-controlled amplification.
- **Unexpected routes.** Unknown method and path combinations return 404 and are recorded in
  the per-route metrics.
- **Preference poisoning (conceptual).** In a real deployment the preference data is the attack
  surface: an adversary who controls comparisons can steer the implicit reward. The evaluation
  measures the implicit reward against ground truth, which is the natural place to detect such
  drift; the data card notes the annotator-noise assumptions.

## Non-goals
- Authentication, authorization, and transport security are delegated to the deployment
  environment (a reverse proxy or gateway), as described in the operations guide.
