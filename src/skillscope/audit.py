"""Read-only source-workbook profiling for SkillScope India."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from skillscope.taxonomy import ROLE_LABELS, match_title
from skillscope.source import EXPECTED_COLUMNS, EXPECTED_SHEET, sha256_file


def normalize_text(series: pd.Series) -> pd.Series:
    """Trim and collapse whitespace while preserving nullable strings."""
    return series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)


def top_values(series: pd.Series, limit: int = 10) -> list[dict[str, Any]]:
    counts = normalize_text(series).replace("", pd.NA).value_counts(dropna=True).head(limit)
    return [{"value": str(value), "count": int(count)} for value, count in counts.items()]


def column_profile(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = len(df)
    if rows == 0:
        raise ValueError("Cannot profile an empty dataset")

    output: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        blank_count = 0
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            blank_count = int(series.astype("string").str.strip().eq("").fillna(False).sum())
        missing_count = int(series.isna().sum()) + blank_count
        output.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing_or_blank": missing_count,
                "missing_or_blank_pct": round(100 * missing_count / rows, 3),
                "unique_non_null": int(series.nunique(dropna=True)),
            }
        )
    return output


def build_audit(source: Path) -> dict[str, Any]:
    """Read an untouched workbook and return a JSON-serializable quality report."""
    with pd.ExcelFile(source) as workbook:
        sheet_names = workbook.sheet_names
    if sheet_names != [EXPECTED_SHEET]:
        raise ValueError(f"Expected exactly ['{EXPECTED_SHEET}']; found {sheet_names}")

    df = pd.read_excel(source, sheet_name=EXPECTED_SHEET)
    if tuple(df.columns) != EXPECTED_COLUMNS:
        raise ValueError("Source schema differs from the expected 17-column contract")

    rows = len(df)
    title = normalize_text(df["title"])
    company = normalize_text(df["companyName"])
    location = normalize_text(df["location"])
    descriptions = normalize_text(df["jobDescription"]).str.lower()
    skills = normalize_text(df["tagsAndSkills"])
    salary = normalize_text(df["salary"]).str.lower()
    minimum_salary = pd.to_numeric(df["minimumSalary"], errors="coerce")
    maximum_salary = pd.to_numeric(df["maximumSalary"], errors="coerce")
    minimum_experience = pd.to_numeric(df["minimumExperience"], errors="coerce")
    maximum_experience = pd.to_numeric(df["maximumExperience"], errors="coerce")

    role_matches = title.map(match_title)
    role_masks = {
        role: role_matches.map(lambda item, label=role: label in item.matched_roles)
        for role in ROLE_LABELS
    }
    role_union = pd.concat(role_masks.values(), axis=1).any(axis=1)

    return {
        "source": {
            "file": source.name,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "sheets": sheet_names,
        },
        "shape": {"rows": rows, "columns": len(df.columns)},
        "columns": list(df.columns),
        "column_profile": column_profile(df),
        "duplicates": {
            "exact_rows": int(df.duplicated().sum()),
            "duplicate_job_id_rows_excluding_first": int(df["jobId"].duplicated().sum()),
            "duplicate_normalized_descriptions_excluding_first": int(
                descriptions[descriptions.notna() & descriptions.ne("")].duplicated().sum()
            ),
            "duplicate_title_company_location_excluding_first": int(
                pd.DataFrame(
                    {
                        "title": title.str.lower(),
                        "company": company.str.lower(),
                        "location": location.str.lower(),
                    }
                ).duplicated().sum()
            ),
        },
        "field_quality": {
            "salary_not_disclosed_count": int(salary.eq("not disclosed").sum()),
            "salary_not_disclosed_pct": round(100 * salary.eq("not disclosed").sum() / rows, 3),
            "zero_zero_salary_count": int(((minimum_salary == 0) & (maximum_salary == 0)).sum()),
            "invalid_salary_range_count": int(
                ((minimum_salary > maximum_salary) & minimum_salary.notna() & maximum_salary.notna()).sum()
            ),
            "invalid_experience_range_count": int(
                (
                    (minimum_experience > maximum_experience)
                    & minimum_experience.notna()
                    & maximum_experience.notna()
                ).sum()
            ),
            "entry_level_minimum_experience_0_to_2_count": int(minimum_experience.le(2).sum()),
            "skills_present_count": int(skills.notna().sum() - skills.eq("").sum()),
            "skills_present_pct": round(100 * (skills.notna().sum() - skills.eq("").sum()) / rows, 3),
            "descriptions_containing_html_count": int(
                df["jobDescription"].astype("string").str.contains(r"<[^>]+>", regex=True, na=False).sum()
            ),
            "descriptions_containing_phone_like_count": int(
                df["jobDescription"]
                .astype("string")
                .str.contains(r"(?<!\d)[6-9]\d{9}(?!\d)", regex=True, na=False)
                .sum()
            ),
            "relative_posting_date_values": top_values(df["jobUploaded"], limit=40),
        },
        "cardinality": {
            "unique_job_ids": int(df["jobId"].nunique(dropna=True)),
            "unique_titles_normalized": int(title.str.lower().nunique(dropna=True)),
            "unique_companies_normalized": int(company.str.lower().nunique(dropna=True)),
            "unique_locations_normalized": int(location.str.lower().nunique(dropna=True)),
        },
        "candidate_role_filter": {
            "counts_nonexclusive": {role: int(mask.sum()) for role, mask in role_masks.items()},
            "entry_level_counts_nonexclusive": {
                role: int((mask & minimum_experience.le(2)).sum()) for role, mask in role_masks.items()
            },
            "union_count": int(role_union.sum()),
            "union_pct": round(100 * role_union.sum() / rows, 3),
        },
        "top_values": {
            "titles": top_values(df["title"]),
            "companies": top_values(df["companyName"]),
            "locations": top_values(df["location"]),
            "currencies": top_values(df["currency"]),
        },
    }


def write_audit(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
