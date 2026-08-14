"""Run and validate the tracked DuckDB analytical queries."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


QUERY_FILES = (
    "01_role_distribution.sql",
    "02_top_skills_overall.sql",
    "03_top_skills_by_role.sql",
    "04_entry_level_skills.sql",
    "05_city_role_distribution.sql",
    "06_skill_cooccurrence.sql",
    "07_company_concentration.sql",
)

EXPECTED_COLUMNS = {
    "role_distribution": {"role_category", "posting_count", "share_pct"},
    "top_skills_overall": {"skill_rank", "skill", "posting_count", "denominator", "penetration_pct"},
    "top_skills_by_role": {"role_category", "skill_rank", "skill", "posting_count", "denominator", "penetration_pct"},
    "entry_level_skills": {"skill_rank", "skill", "posting_count", "denominator", "penetration_pct"},
    "city_role_distribution": {"primary_city", "city_postings", "role_category", "role_postings", "within_city_pct"},
    "skill_cooccurrence": {"pair_rank", "skill_a", "skill_b", "pair_posting_count", "skill_a_postings", "skill_b_postings", "denominator", "pair_penetration_pct", "lift"},
    "company_concentration": {"company_rank", "company_name", "posting_count", "represented_roles", "denominator", "share_pct"},
}


def query_name(path: Path) -> str:
    """Convert a numbered SQL filename into a stable result name."""
    return path.stem.split("_", 1)[1]


def create_parquet_views(connection: duckdb.DuckDBPyConnection, jobs_path: Path, skills_path: Path) -> None:
    """Expose local Parquet files as read-only analytical views."""
    jobs_sql = str(jobs_path.resolve()).replace("'", "''")
    skills_sql = str(skills_path.resolve()).replace("'", "''")
    connection.execute(f"CREATE OR REPLACE VIEW jobs AS SELECT * FROM read_parquet('{jobs_sql}')")
    connection.execute(f"CREATE OR REPLACE VIEW job_skills AS SELECT * FROM read_parquet('{skills_sql}')")


def run_queries(connection: duckdb.DuckDBPyConnection, sql_dir: Path) -> dict[str, pd.DataFrame]:
    """Execute the versioned queries in their declared order."""
    results: dict[str, pd.DataFrame] = {}
    for filename in QUERY_FILES:
        path = sql_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        results[query_name(path)] = connection.execute(path.read_text(encoding="utf-8")).df()
    return results


def validate_results(results: dict[str, pd.DataFrame]) -> list[str]:
    """Return structural or metric violations across analysis outputs."""
    errors: list[str] = []
    if set(results) != set(EXPECTED_COLUMNS):
        errors.append("analysis result set does not match the declared queries")
        return errors
    for name, frame in results.items():
        missing = EXPECTED_COLUMNS[name] - set(frame.columns)
        if missing:
            errors.append(f"{name} is missing columns: {sorted(missing)}")
        if frame.empty:
            errors.append(f"{name} is empty")
    for name, percentage in (
        ("role_distribution", "share_pct"),
        ("top_skills_overall", "penetration_pct"),
        ("top_skills_by_role", "penetration_pct"),
        ("entry_level_skills", "penetration_pct"),
        ("city_role_distribution", "within_city_pct"),
        ("skill_cooccurrence", "pair_penetration_pct"),
        ("company_concentration", "share_pct"),
    ):
        if percentage in results[name] and not results[name][percentage].between(0, 100).all():
            errors.append(f"{name}.{percentage} must be between 0 and 100")
    if "share_pct" in results["role_distribution"] and abs(results["role_distribution"]["share_pct"].sum() - 100) > 0.05:
        errors.append("rounded role shares do not sum to approximately 100")
    if "lift" in results["skill_cooccurrence"] and (results["skill_cooccurrence"]["lift"] <= 0).any():
        errors.append("co-occurrence lift must be positive")
    return errors
