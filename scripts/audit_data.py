#!/usr/bin/env python3
"""Generate the SkillScope source-data audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from skillscope.audit import build_audit, write_audit  # noqa: E402
from skillscope.source import (  # noqa: E402
    SourceContractError,
    resolve_source_workbook,
    validate_source_workbook,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "generated" / "data_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the untouched SkillScope source workbook.")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source = resolve_source_workbook(PROJECT_ROOT / "data" / "raw", args.source)
        validate_source_workbook(source)
    except SourceContractError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    report = build_audit(source)
    write_audit(report, args.output)
    roles = report["candidate_role_filter"]
    print(f"PASS: audited {report['shape']['rows']:,} rows and {report['shape']['columns']} columns")
    print(f"SHA-256: {report['source']['sha256']}")
    print(f"Candidate data-role postings: {roles['union_count']:,} ({roles['union_pct']:.3f}%)")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
