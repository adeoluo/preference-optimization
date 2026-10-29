"""Command line interface for preference optimization."""

from __future__ import annotations

import argparse

from preference_optimization.pipeline import PipelineConfig, run_pipeline


def _cmd_evaluate(args: argparse.Namespace) -> None:
    result = run_pipeline(PipelineConfig(beta=args.beta, seed=args.seed))
    print("Direct Preference Optimization, aligned policy versus reference:")
    print(f"  {result.report.format()}")


def _cmd_compare(args: argparse.Namespace) -> None:
    result = run_pipeline(PipelineConfig(beta=args.beta, seed=args.seed))
    domain = result.domain
    print("Per-prompt greedy choice (reference -> aligned, with the gold-best response):")
    for prompt in range(domain.n_prompts):
        reference_choice = result.reference.greedy(prompt)
        policy_choice = result.policy.greedy(prompt)
        best = domain.best_response(prompt)
        marker = "match" if policy_choice == best else "miss"
        print(
            f"  prompt {prompt}: reference r{reference_choice} "
            f"-> aligned r{policy_choice}  (gold-best r{best}, {marker})"
        )


def _cmd_serve(args: argparse.Namespace) -> None:
    from preference_optimization.service import serve

    serve(host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="preference-optimization", description=__doc__)
    parser.add_argument("--beta", type=float, default=0.3, help="DPO regularization strength")
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("evaluate", help="run the full pipeline and print the alignment report")
    sub.add_parser("compare", help="show the greedy choice per prompt before and after DPO")

    serve = sub.add_parser("serve", help="start the HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    handlers = {"evaluate": _cmd_evaluate, "compare": _cmd_compare, "serve": _cmd_serve}
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()
