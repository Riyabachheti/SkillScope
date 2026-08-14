"""Validation contracts for SkillScope processed tables."""

from __future__ import annotations

import pandas as pd

from skillscope.taxonomy import ROLE_LABELS
from skillscope.transform import EMAIL_PATTERN, HTML_TAG_PATTERN, PHONE_PATTERN


JOBS_REQUIRED_COLUMNS = {
    "job_id",
    "role_category",
    "title_original",
    "title_normalized",
    "company_id",
    "company_name",
    "company_key",
    "location_original",
    "primary_city",
    "work_arrangement",
    "location_count",
    "minimum_experience",
    "maximum_experience",
    "experience_band",
    "salary_original",
    "salary_disclosed",
    "minimum_salary",
    "maximum_salary",
    "currency",
    "job_uploaded_relative",
    "reviews_count",
    "aggregate_rating",
    "description_clean",
    "template_group_id",
}
JOB_SKILLS_REQUIRED_COLUMNS = {"job_id", "skill_key", "skill"}
TRANSFORMATION_COUNT_KEYS = {
    "source_rows",
    "matched_postings",
    "collision_postings_withheld",
    "rule_labelled_postings",
    "reviewed_near_misses_added",
    "reviewed_role_corrections_applied",
    "reviewed_title_overrides_applied",
    "unambiguous_labelled_postings",
    "duplicate_job_ids_removed",
    "exact_cross_id_duplicates_removed",
    "retained_jobs",
    "job_skill_pairs",
    "jobs_without_skills",
    "template_groups",
    "repeated_template_groups",
    "jobs_in_repeated_template_groups",
    "largest_template_group_size",
    "role_counts",
}


def validate_processed_tables(jobs: pd.DataFrame, job_skills: pd.DataFrame) -> list[str]:
    """Return every contract violation instead of stopping at the first one."""
    errors: list[str] = []
    missing_jobs = sorted(JOBS_REQUIRED_COLUMNS - set(jobs.columns))
    missing_skills = sorted(JOB_SKILLS_REQUIRED_COLUMNS - set(job_skills.columns))
    if missing_jobs:
        errors.append(f"jobs is missing required columns: {missing_jobs}")
    if missing_skills:
        errors.append(f"job_skills is missing required columns: {missing_skills}")
    if missing_jobs or missing_skills:
        return errors
    if jobs.empty:
        errors.append("jobs table is empty")
        return errors
    if jobs["job_id"].isna().any() or jobs["job_id"].duplicated().any():
        errors.append("job_id must be non-null and unique")
    unknown_roles = set(jobs["role_category"].dropna()) - set(ROLE_LABELS)
    if unknown_roles or jobs["role_category"].isna().any():
        errors.append(f"role_category contains invalid values: {sorted(unknown_roles)}")
    if (jobs["location_count"] < 1).any():
        errors.append("location_count must be at least one")
    if (~jobs["work_arrangement"].isin(["Remote", "Hybrid", "Not specified"])).any():
        errors.append("work_arrangement contains invalid values")
    template_ids = jobs["template_group_id"]
    if template_ids.isna().any() or not template_ids.astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        errors.append("template_group_id must be a 64-character lowercase SHA-256 value")

    both_experience = jobs[["minimum_experience", "maximum_experience"]].notna().all(axis=1)
    if (jobs.loc[both_experience, "minimum_experience"] > jobs.loc[both_experience, "maximum_experience"]).any():
        errors.append("minimum_experience exceeds maximum_experience")
    if (jobs["minimum_experience"].dropna() < 0).any() or (jobs["maximum_experience"].dropna() < 0).any():
        errors.append("experience cannot be negative")

    both_salary = jobs[["minimum_salary", "maximum_salary"]].notna().all(axis=1)
    if (jobs.loc[both_salary, "minimum_salary"] > jobs.loc[both_salary, "maximum_salary"]).any():
        errors.append("minimum_salary exceeds maximum_salary")
    if jobs.loc[~jobs["salary_disclosed"], ["minimum_salary", "maximum_salary"]].notna().any(axis=None):
        errors.append("undisclosed salary rows must not expose numeric salary values")

    descriptions = jobs["description_clean"].fillna("").astype(str)
    if descriptions.str.contains(HTML_TAG_PATTERN).any():
        errors.append("description_clean still contains HTML tags")
    if descriptions.str.contains(EMAIL_PATTERN).any():
        errors.append("description_clean still contains email addresses")
    if descriptions.str.contains(PHONE_PATTERN).any():
        errors.append("description_clean still contains phone numbers")

    if job_skills[["job_id", "skill_key"]].duplicated().any():
        errors.append("job_skills contains duplicate job_id/skill_key pairs")
    orphan_ids = set(job_skills["job_id"]) - set(jobs["job_id"])
    if orphan_ids:
        errors.append(f"job_skills contains {len(orphan_ids)} orphan job IDs")
    if job_skills["skill_key"].isna().any() or job_skills["skill_key"].astype(str).str.strip().eq("").any():
        errors.append("skill_key must be non-empty")
    return errors


def validate_transformation_counts(
    metrics: dict[str, object],
    jobs: pd.DataFrame,
    job_skills: pd.DataFrame,
) -> list[str]:
    """Check that every Phase 3 count stage balances with the written tables."""
    errors: list[str] = []
    missing = sorted(TRANSFORMATION_COUNT_KEYS - set(metrics))
    if missing:
        return [f"transformation metrics are missing required keys: {missing}"]

    def count(name: str) -> int:
        return int(metrics[name])

    if count("matched_postings") - count("collision_postings_withheld") != count(
        "rule_labelled_postings"
    ):
        errors.append("rule-labelled count does not equal matched minus collisions")
    if count("rule_labelled_postings") + count("reviewed_near_misses_added") != count(
        "unambiguous_labelled_postings"
    ):
        errors.append("labelled count does not equal rule labels plus reviewed near misses")
    if count("reviewed_near_misses_added") + count(
        "reviewed_role_corrections_applied"
    ) != count("reviewed_title_overrides_applied"):
        errors.append("reviewed override count does not equal additions plus corrections")
    if count("unambiguous_labelled_postings") - count(
        "duplicate_job_ids_removed"
    ) - count("exact_cross_id_duplicates_removed") != count("retained_jobs"):
        errors.append("retained count does not equal labelled minus both duplicate stages")
    if count("retained_jobs") != len(jobs):
        errors.append("retained_jobs does not equal the jobs table row count")
    if count("job_skill_pairs") != len(job_skills):
        errors.append("job_skill_pairs does not equal the job_skills table row count")

    jobs_without_skills = len(set(jobs["job_id"]) - set(job_skills["job_id"]))
    if count("jobs_without_skills") != jobs_without_skills:
        errors.append("jobs_without_skills does not match the processed tables")

    role_counts = metrics["role_counts"]
    if not isinstance(role_counts, dict) or set(role_counts) != set(ROLE_LABELS):
        errors.append("role_counts must contain exactly the four role labels")
    elif sum(int(value) for value in role_counts.values()) != count("retained_jobs"):
        errors.append("role_counts do not sum to retained_jobs")

    template_sizes = jobs["template_group_id"].value_counts()
    repeated_sizes = template_sizes[template_sizes > 1]
    expected_template_values = {
        "template_groups": len(template_sizes),
        "repeated_template_groups": len(repeated_sizes),
        "jobs_in_repeated_template_groups": int(repeated_sizes.sum()),
        "largest_template_group_size": int(template_sizes.max()),
    }
    for name, expected in expected_template_values.items():
        if count(name) != expected:
            errors.append(f"{name} does not match the jobs table")

    if count("source_rows") < count("matched_postings"):
        errors.append("matched_postings cannot exceed source_rows")
    return errors
