#!/usr/bin/env python3
"""Generate the deterministic Phase 2 title-review sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from skillscope.review import build_review_sample, write_json  # noqa: E402
from skillscope.source import (  # noqa: E402
    EXPECTED_SHEET,
    SourceContractError,
    resolve_source_workbook,
    validate_source_workbook,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "generated" / "role_review_sample.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic title-review strata.")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-per-role", type=int, default=30)
    parser.add_argument("--extra-sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source = resolve_source_workbook(PROJECT_ROOT / "data" / "raw", args.source)
        validate_source_workbook(source)
    except SourceContractError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    frame = pd.read_excel(
        source,
        sheet_name=EXPECTED_SHEET,
        usecols=["jobId", "title", "companyName", "tagsAndSkills"],
        engine="openpyxl",
    )
    sample = build_review_sample(
        frame,
        sample_per_role=args.sample_per_role,
        extra_sample_size=args.extra_sample_size,
        seed=args.seed,
    )
    write_json(sample, args.output)
    counts: dict[str, int] = {}
    for item in sample:
        key = item["predicted_role"] or item["stratum"]
        counts[key] = counts.get(key, 0) + 1
    print(f"PASS: wrote {len(sample)} unique-title review records")
    for key, count in counts.items():
        print(f"- {key}: {count}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
