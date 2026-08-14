#!/usr/bin/env python3
"""Execute all Phase 4 SQL and write deterministic local result artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skillscope.analysis import create_parquet_views, run_queries, validate_results  # noqa: E402
from skillscope.transform import file_sha256  # noqa: E402


def main() -> int:
    jobs_path = ROOT / "data" / "processed" / "jobs.parquet"
    skills_path = ROOT / "data" / "processed" / "job_skills.parquet"
    if not jobs_path.is_file() or not skills_path.is_file():
        print("Processed Parquet files are missing; run build_processed_data.py first.", file=sys.stderr)
        return 1

    connection = duckdb.connect(database=":memory:")
    create_parquet_views(connection, jobs_path, skills_path)
    results = run_queries(connection, ROOT / "sql")
    errors = validate_results(results)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    output_dir = ROOT / "reports" / "generated" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "inputs": {
            "jobs_sha256": file_sha256(jobs_path),
            "job_skills_sha256": file_sha256(skills_path),
        },
        "results": {},
    }
    for name, frame in results.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, lineterminator="\n")
        summary["results"][name] = json.loads(frame.to_json(orient="records"))
    summary_path = ROOT / "reports" / "generated" / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: wrote {len(results)} verified SQL result sets to {output_dir}")
    for name, frame in results.items():
        print(f"- {name}: {len(frame)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
