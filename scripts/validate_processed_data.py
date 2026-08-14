#!/usr/bin/env python3
"""Validate Phase 3 files and query them through DuckDB."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skillscope.contracts import (  # noqa: E402
    validate_processed_tables,
    validate_transformation_counts,
)


def main() -> int:
    jobs_path = ROOT / "data" / "processed" / "jobs.parquet"
    skills_path = ROOT / "data" / "processed" / "job_skills.parquet"
    metrics_path = ROOT / "reports" / "generated" / "transformation_metrics.json"
    if not jobs_path.exists() or not skills_path.exists() or not metrics_path.exists():
        print("Processed Parquet files or transformation metrics are missing; run build_processed_data.py first.", file=sys.stderr)
        return 1
    jobs = pd.read_parquet(jobs_path, engine="pyarrow")
    job_skills = pd.read_parquet(skills_path, engine="pyarrow")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    errors = validate_processed_tables(jobs, job_skills)
    if not errors:
        errors.extend(validate_transformation_counts(metrics, jobs, job_skills))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    connection = duckdb.connect(database=":memory:")
    jobs_sql = str(jobs_path).replace("'", "''")
    skills_sql = str(skills_path).replace("'", "''")
    job_count, unique_job_count = connection.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT job_id) FROM read_parquet('{jobs_sql}')"
    ).fetchone()
    orphan_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{skills_sql}') s
        LEFT JOIN read_parquet('{jobs_sql}') j USING (job_id)
        WHERE j.job_id IS NULL
        """
    ).fetchone()[0]
    role_count = connection.execute(
        f"SELECT COUNT(DISTINCT role_category) FROM read_parquet('{jobs_sql}')"
    ).fetchone()[0]
    if job_count != unique_job_count or orphan_count != 0 or role_count != 4:
        print("DuckDB smoke-test invariants failed.", file=sys.stderr)
        return 1
    print(
        f"PASS: {job_count} unique jobs, {len(job_skills)} unique job-skill pairs, "
        f"{role_count} roles, 0 orphan skills."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
