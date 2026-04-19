"""CLI entry point for the Frontier Physics Research Agent."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from agent_ultra import __version__
from agent_ultra.core.config import load_config
from agent_ultra.core.session import ResearchSession, TaskCategory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-ultra",
        description="Autonomous Frontier Mathematics and Physics Research Agent",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    sub = parser.add_subparsers(dest="command")

    # --- evaluate -----------------------------------------------------------
    eval_p = sub.add_parser("evaluate", help="Evaluate a theoretical claim")
    eval_p.add_argument("statement", help="The claim to evaluate")
    eval_p.add_argument(
        "--category",
        choices=[c.value for c in TaskCategory],
        default="physics",
    )

    # --- config -------------------------------------------------------------
    cfg_p = sub.add_parser("config", help="Show the active configuration")
    cfg_p.add_argument(
        "--format",
        choices=["yaml", "json"],
        default="yaml",
    )

    # --- constants ----------------------------------------------------------
    const_p = sub.add_parser("constants", help="Look up a physical constant")
    const_p.add_argument("name", help="Constant key (e.g. 'alpha', 'c', 'hbar')")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    if args.command == "evaluate":
        session = ResearchSession(config=config)
        category = TaskCategory(args.category)
        session.start(args.statement, category=category)
        result = session.evaluate_claim(args.statement)
        session.complete()

        print(json.dumps({
            "claim_id": result.claim_id,
            "quality_gates": result.passed_quality_gates,
            "session": session.summary(),
        }, indent=2, default=str))

    elif args.command == "config":
        if args.format == "yaml":
            print(config.to_yaml())
        else:
            print(json.dumps(config.model_dump(mode="json"), indent=2))

    elif args.command == "constants":
        from agent_ultra.knowledge.constants import get_constant

        const = get_constant(args.name)
        if const is None:
            print(f"Unknown constant: {args.name}", file=sys.stderr)
            return 1
        print(json.dumps({
            "name": const.name,
            "symbol": const.symbol,
            "value": const.value,
            "uncertainty": const.uncertainty,
            "unit": const.unit,
            "source": const.source,
        }, indent=2))

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
