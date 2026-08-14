#!/usr/bin/env python3
"""Independently reconcile every Phase 4 SQL output with pandas."""

from __future__ import annotations

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "generated" / "analysis"


def compare(name: str, expected: pd.DataFrame) -> None:
    actual = pd.read_csv(OUTPUT_DIR / f"{name}.csv")
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        atol=1e-9,
        rtol=1e-9,
    )


def role_distribution(jobs: pd.DataFrame) -> pd.DataFrame:
    result = jobs.groupby("role_category").size().rename("posting_count").reset_index()
    result["share_pct"] = (100 * result["posting_count"] / len(jobs)).round(2)
    return result.sort_values(["posting_count", "role_category"], ascending=[False, True])


def top_skills(jobs: pd.DataFrame, skills: pd.DataFrame) -> pd.DataFrame:
    result = skills.groupby("skill_key").agg(skill=("skill", "min"), posting_count=("job_id", "size")).reset_index()
    result = result.sort_values(["posting_count", "skill_key"], ascending=[False, True]).head(20)
    result.insert(0, "skill_rank", range(1, len(result) + 1))
    result["denominator"] = len(jobs)
    result["penetration_pct"] = (100 * result["posting_count"] / len(jobs)).round(2)
    return result[["skill_rank", "skill", "posting_count", "denominator", "penetration_pct"]]


def role_skills(jobs: pd.DataFrame, skills: pd.DataFrame) -> pd.DataFrame:
    merged = skills.merge(jobs[["job_id", "role_category"]], on="job_id", validate="many_to_one")
    result = merged.groupby(["role_category", "skill_key"]).agg(
        skill=("skill", "min"), posting_count=("job_id", "size")
    ).reset_index()
    result = result.sort_values(
        ["role_category", "posting_count", "skill_key"], ascending=[True, False, True]
    )
    result["skill_rank"] = result.groupby("role_category").cumcount() + 1
    result = result[result["skill_rank"] <= 10].copy()
    denominators = jobs.groupby("role_category").size()
    result["denominator"] = result["role_category"].map(denominators)
    result["penetration_pct"] = (100 * result["posting_count"] / result["denominator"]).round(2)
    return result[["role_category", "skill_rank", "skill", "posting_count", "denominator", "penetration_pct"]]


def entry_skills(jobs: pd.DataFrame, skills: pd.DataFrame) -> pd.DataFrame:
    entry_ids = jobs.loc[jobs["experience_band"] == "Entry level (0-2 years)", "job_id"]
    selected = skills[skills["job_id"].isin(entry_ids)]
    result = selected.groupby("skill_key").agg(skill=("skill", "min"), posting_count=("job_id", "size")).reset_index()
    result = result.sort_values(["posting_count", "skill_key"], ascending=[False, True]).head(20)
    result.insert(0, "skill_rank", range(1, len(result) + 1))
    result["denominator"] = len(entry_ids)
    result["penetration_pct"] = (100 * result["posting_count"] / len(entry_ids)).round(2)
    return result[["skill_rank", "skill", "posting_count", "denominator", "penetration_pct"]]


def city_roles(jobs: pd.DataFrame) -> pd.DataFrame:
    physical = jobs[jobs["primary_city"] != "Remote"]
    totals = physical.groupby("primary_city").size().rename("city_postings").reset_index()
    top = totals.sort_values(["city_postings", "primary_city"], ascending=[False, True]).head(10)
    selected = physical[physical["primary_city"].isin(top["primary_city"])]
    result = selected.groupby(["primary_city", "role_category"]).size().rename("role_postings").reset_index()
    result = result.merge(top, on="primary_city", validate="many_to_one")
    result["within_city_pct"] = (100 * result["role_postings"] / result["city_postings"]).round(2)
    result = result.sort_values(
        ["city_postings", "primary_city", "role_postings", "role_category"],
        ascending=[False, True, False, True],
    )
    return result[["primary_city", "city_postings", "role_category", "role_postings", "within_city_pct"]]


def skill_pairs(jobs: pd.DataFrame, skills: pd.DataFrame) -> pd.DataFrame:
    pair_counts: Counter[tuple[str, str]] = Counter()
    for values in skills.groupby("job_id")["skill_key"]:
        pair_counts.update(combinations(sorted(values[1]), 2))
    skill_counts = skills.groupby("skill_key").size().to_dict()
    displays = skills.groupby("skill_key")["skill"].min().to_dict()
    ordered = sorted(pair_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))[:20]
    rows = []
    for rank, ((skill_a, skill_b), pair_count) in enumerate(ordered, start=1):
        count_a, count_b = skill_counts[skill_a], skill_counts[skill_b]
        rows.append(
            {
                "pair_rank": rank,
                "skill_a": displays[skill_a],
                "skill_b": displays[skill_b],
                "pair_posting_count": pair_count,
                "skill_a_postings": count_a,
                "skill_b_postings": count_b,
                "denominator": len(jobs),
                "pair_penetration_pct": round(100 * pair_count / len(jobs), 2),
                "lift": round(pair_count * len(jobs) / (count_a * count_b), 3),
            }
        )
    return pd.DataFrame(rows)


def company_concentration(jobs: pd.DataFrame) -> pd.DataFrame:
    result = jobs.groupby("company_key").agg(
        company_name=("company_name", "min"),
        posting_count=("job_id", "size"),
        represented_roles=("role_category", "nunique"),
    ).reset_index()
    result = result.sort_values(["posting_count", "company_key"], ascending=[False, True]).head(15)
    result.insert(0, "company_rank", range(1, len(result) + 1))
    result["denominator"] = len(jobs)
    result["share_pct"] = (100 * result["posting_count"] / len(jobs)).round(2)
    return result[["company_rank", "company_name", "posting_count", "represented_roles", "denominator", "share_pct"]]


def main() -> int:
    jobs = pd.read_parquet(ROOT / "data" / "processed" / "jobs.parquet")
    skills = pd.read_parquet(ROOT / "data" / "processed" / "job_skills.parquet")
    expected = {
        "role_distribution": role_distribution(jobs),
        "top_skills_overall": top_skills(jobs, skills),
        "top_skills_by_role": role_skills(jobs, skills),
        "entry_level_skills": entry_skills(jobs, skills),
        "city_role_distribution": city_roles(jobs),
        "skill_cooccurrence": skill_pairs(jobs, skills),
        "company_concentration": company_concentration(jobs),
    }
    try:
        for name, frame in expected.items():
            compare(name, frame)
    except (AssertionError, FileNotFoundError) as error:
        print(f"ERROR: analysis reconciliation failed: {error}", file=sys.stderr)
        return 1
    print("PASS: all 7 DuckDB result sets reconcile exactly with independent pandas calculations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
