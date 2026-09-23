"""
Stage 10 – 100-turn benchmark + regression gate

Convenience wrapper around performance.py with defaults suited for CI:

  PYTHONPATH=backend:. python -m evals.benchmark
  PYTHONPATH=backend:. python -m evals.benchmark --turns 100
  PYTHONPATH=backend:. python -m evals.benchmark --update-baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.performance import main as perf_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="100-turn voice agent benchmark")
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--update-baseline", action="store_true",
                        help="Write results to evals/baseline.json after the run")
    parser.add_argument("--baseline", type=str, default=str(ROOT / "evals" / "baseline.json"))
    args, unknown = parser.parse_known_args(argv)

    perf_argv = ["--turns", str(args.turns), "--baseline", args.baseline]
    if args.update_baseline:
        perf_argv += ["--save-baseline", args.baseline]
    # Forward any extra flags
    perf_argv += unknown

    print("=" * 56)
    print("  Stage 10 – 100-turn benchmark + regression gate")
    print("=" * 56)
    return perf_main(perf_argv)


if __name__ == "__main__":
    raise SystemExit(main())
