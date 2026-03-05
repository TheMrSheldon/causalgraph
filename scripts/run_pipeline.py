#!/usr/bin/env python
"""
CLI entry point for the r/science causal pipeline.

Usage:
    python scripts/run_pipeline.py                     # Run all steps
    python scripts/run_pipeline.py --step 1            # Only step 1
    python scripts/run_pipeline.py --config my.yaml    # Custom config
"""
import argparse
import sys
from pathlib import Path

# Make sure the workspace root is on the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.runner import run_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="r/science causal relationship extraction pipeline"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run only a specific step (1=identify, 2=extract, 3=cluster). Omit to run all.",
    )
    args = parser.parse_args()
    run_all(config_path=args.config, step=args.step)


if __name__ == "__main__":
    main()
