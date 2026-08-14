#!/usr/bin/env python3
"""Build the Phase 3 Parquet tables from the untouched source workbook."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skillscope.contracts import (  # noqa: E402
    validate_processed_tables,
    validate_transformation_counts,
)
from skillscope.review import (  # noqa: E402
    ReviewGateError,
    build_review_sample,
    validate_review_gate,
)
from skillscope.source import (  # noqa: E402
    SourceContractError,
    resolve_source_workbook,
    sha256_file,
    validate_source_workbook,
)
from skillscope.transform import build_processed_tables, load_aliases  # noqa: E402


def main() -> int:
    try:
        source_path = resolve_source_workbook(ROOT / "data" / "raw")
        source_identity = validate_source_workbook(source_path)
    except SourceContractError as error:
        print(f"Source validation failed: {error}", file=sys.stderr)
        return 2

    source = pd.read_excel(
        source_path,
        sheet_name=source_identity.sheet,
        engine="openpyxl",
    )
    expected_review_sample = build_review_sample(
        source[["jobId", "title", "companyName", "tagsAndSkills"]]
    )
    try:
        review_metrics = validate_review_gate(
            ROOT / "reports" / "generated" / "role_review_sample.json",
            ROOT / "data" / "reference" / "role_review_annotations.json",
            expected_sample=expected_review_sample,
        )
    except ReviewGateError as error:
        print(f"Role-review validation failed: {error}", file=sys.stderr)
        return 3

    jobs, job_skills, metrics = build_processed_tables(
        source,
        load_aliases(ROOT / "data" / "reference" / "location_aliases.json"),
        load_aliases(ROOT / "data" / "reference" / "skill_aliases.json"),
    )
    errors = validate_processed_tables(jobs, job_skills)
    if not errors:
        errors.extend(validate_transformation_counts(metrics, jobs, job_skills))
    if errors:
        print("Processed-data validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    output_dir = ROOT / "data" / "processed"
    report_dir = ROOT / "reports" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = output_dir / "jobs.parquet"
    skills_path = output_dir / "job_skills.parquet"
    jobs.to_parquet(jobs_path, engine="pyarrow", compression="snappy", index=False)
    job_skills.to_parquet(skills_path, engine="pyarrow", compression="snappy", index=False)
    metrics["source_file"] = source_path.name
    metrics["source_sha256"] = source_identity.sha256
    metrics["role_review_sample_sha256"] = review_metrics["sample_sha256"]
    metrics["role_review_count"] = review_metrics["reviewed_count"]
    metrics["role_review_passed"] = True
    metrics["jobs_sha256"] = sha256_file(jobs_path)
    metrics["job_skills_sha256"] = sha256_file(skills_path)
    (report_dir / "transformation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
