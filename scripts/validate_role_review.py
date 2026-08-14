#!/usr/bin/env python3
"""Calculate Phase 2 taxonomy quality from independent review labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from skillscope.review import ReviewGateError, validate_review_gate, write_json  # noqa: E402


DEFAULT_SAMPLE = PROJECT_ROOT / "reports" / "generated" / "role_review_sample.json"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "reference" / "role_review_annotations.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "generated" / "role_validation_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate independently reviewed role labels.")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = validate_review_gate(
            args.sample,
            args.labels,
            minimum_precision=args.minimum_precision,
        )
    except ReviewGateError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 3

    write_json(metrics, args.output)
    print(f"Reviewed: {metrics['reviewed_count']}/{metrics['sample_count']}")
    for role, result in metrics["per_role"].items():
        print(f"- {role}: {result['correct']}/{result['reviewed']} ({result['precision']:.1%})")
    print(f"Near misses judged in scope: {metrics['near_miss_in_scope']}/{metrics['near_miss_reviewed']}")
    print(f"Report: {args.output}")

    print("PASS: role-review completeness and precision thresholds satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
