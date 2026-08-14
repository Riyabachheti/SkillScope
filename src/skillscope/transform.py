"""Deterministic transformation of the audited workbook into analytical tables."""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from skillscope.taxonomy import (
    REVIEWED_TITLE_OVERRIDES,
    ROLE_LABELS,
    apply_reviewed_title_override,
    match_title,
    normalize_title,
)


EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9](?:[\s-]?\d){9}(?!\d)")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
BREAK_PATTERN = re.compile(r"<\s*(?:br|/p|/li|/div|/h[1-6])\s*/?\s*>", re.IGNORECASE)
SPACE_PATTERN = re.compile(r"\s+")
PARENTHETICAL_PATTERN = re.compile(r"\s*\([^)]*\)\s*")

# A different job ID is deleted only when every other source field is equal.
# Keeping this list explicit makes the deletion rule easy to review and test.
EXACT_DUPLICATE_FIELDS = (
    "title",
    "currency",
    "jobUploaded",
    "companyName",
    "tagsAndSkills",
    "experience",
    "salary",
    "location",
    "companyId",
    "ReviewsCount",
    "AggregateRating",
    "jobDescription",
    "minimumSalary",
    "maximumSalary",
    "minimumExperience",
    "maximumExperience",
)


@dataclass(frozen=True)
class Location:
    primary_city: str
    work_arrangement: str
    location_count: int


def normalized_text(value: object) -> str:
    """Normalize Unicode and whitespace while preserving readable casing."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return SPACE_PATTERN.sub(" ", text).strip()


def normalized_key(value: object) -> str:
    """Return a stable comparison key."""
    return normalized_text(value).casefold()


def clean_description(value: object) -> str:
    """Remove HTML, normalize whitespace, and redact common contact details."""
    text = html.unescape(normalized_text(value))
    text = BREAK_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
    text = PHONE_PATTERN.sub("[PHONE_REDACTED]", text)
    return SPACE_PATTERN.sub(" ", text).strip()


def make_template_group_id(company: object, description: object) -> str:
    """Identify reused company/description templates without deleting their jobs."""
    parts = [normalized_key(company), normalized_key(clean_description(description))]
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_location(value: object, aliases: dict[str, str]) -> Location:
    """Extract a conservative primary city and explicit work arrangement."""
    original = normalized_text(value)
    lower = original.casefold()
    if lower == "remote":
        return Location("Remote", "Remote", 1)

    work_arrangement = "Hybrid" if lower.startswith("hybrid - ") else "Not specified"
    without_mode = re.sub(r"^hybrid\s*-\s*", "", original, flags=re.IGNORECASE)
    parts = [part.strip() for part in without_mode.split(",") if part.strip()]
    cleaned_parts = [PARENTHETICAL_PATTERN.sub("", part).strip() for part in parts]
    cleaned_parts = [part for part in cleaned_parts if part]
    primary = cleaned_parts[0] if cleaned_parts else "Unknown"
    if primary.casefold() == "india" and len(cleaned_parts) > 1:
        primary = cleaned_parts[1]
    primary = aliases.get(primary.casefold(), primary)
    return Location(primary, work_arrangement, max(1, len(cleaned_parts)))


def experience_band(minimum: object) -> str:
    """Bucket a posting by its stated minimum required experience."""
    if pd.isna(minimum):
        return "Unknown"
    years = float(minimum)
    if years <= 2:
        return "Entry level (0-2 years)"
    if years <= 5:
        return "Early career (3-5 years)"
    if years <= 9:
        return "Mid level (6-9 years)"
    return "Senior (10+ years)"


def canonical_skill(value: object, aliases: dict[str, str]) -> tuple[str, str]:
    """Return a normalized comparison key and a readable display label."""
    clean = normalized_text(value)
    key = clean.casefold()
    display = aliases.get(key, clean.title())
    return normalized_key(display), display


def load_aliases(path: Path) -> dict[str, str]:
    """Load and normalize a JSON alias mapping."""
    values = json.loads(path.read_text(encoding="utf-8"))
    return {normalized_key(key): normalized_text(value) for key, value in values.items()}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_processed_tables(
    source: pd.DataFrame,
    location_aliases: dict[str, str],
    skill_aliases: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Create clean job and job-skill tables plus transformation metrics."""
    working = source.copy()
    working["_source_row"] = range(len(working))
    rule_matches = working["title"].map(match_title)
    reviewed_matches = rule_matches.map(apply_reviewed_title_override)
    working["_matched_count"] = rule_matches.map(lambda result: len(result.matched_roles))
    working["role_category"] = reviewed_matches.map(lambda result: result.predicted_role)

    matched_count = int((working["_matched_count"] > 0).sum())
    collision_count = int((working["_matched_count"] > 1).sum())
    rule_labelled_count = int(rule_matches.map(lambda result: result.predicted_role is not None).sum())
    reviewed_near_misses_added = int(
        sum(
            base.predicted_role is None and reviewed.predicted_role is not None
            for base, reviewed in zip(rule_matches, reviewed_matches)
        )
    )
    reviewed_role_corrections = int(
        sum(
            base.predicted_role is not None
            and reviewed.predicted_role != base.predicted_role
            for base, reviewed in zip(rule_matches, reviewed_matches)
        )
    )
    reviewed_override_count = int(
        reviewed_matches.map(lambda result: result.stratum == "reviewed_override").sum()
    )
    labelled = working.loc[working["role_category"].notna()].copy()
    unambiguous_count = len(labelled)

    completeness_fields = [
        "companyName", "tagsAndSkills", "location", "jobDescription",
        "minimumExperience", "maximumExperience", "salary",
    ]
    labelled["_completeness"] = labelled[completeness_fields].notna().sum(axis=1)
    labelled = labelled.sort_values(
        ["jobId", "_completeness", "_source_row"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    before_id_dedup = len(labelled)
    labelled = labelled.drop_duplicates("jobId", keep="first")
    duplicate_job_ids_removed = before_id_dedup - len(labelled)

    labelled = labelled.sort_values("_source_row", kind="mergesort")
    before_exact_dedup = len(labelled)
    labelled = labelled.drop_duplicates(list(EXACT_DUPLICATE_FIELDS), keep="first")
    exact_cross_id_duplicates_removed = before_exact_dedup - len(labelled)

    labelled["title_normalized"] = labelled["title"].map(normalize_title)
    labelled["company_name"] = labelled["companyName"].map(normalized_text)
    labelled["company_key"] = labelled["companyName"].map(normalized_key)
    labelled["location_original"] = labelled["location"].map(normalized_text)
    labelled["description_clean"] = labelled["jobDescription"].map(clean_description)
    labelled["template_group_id"] = labelled.apply(
        lambda row: make_template_group_id(row["companyName"], row["jobDescription"]),
        axis=1,
    )

    locations = labelled["location_original"].map(
        lambda value: parse_location(value, location_aliases)
    )
    labelled["primary_city"] = locations.map(lambda value: value.primary_city)
    labelled["work_arrangement"] = locations.map(lambda value: value.work_arrangement)
    labelled["location_count"] = locations.map(lambda value: value.location_count)

    labelled["minimum_experience"] = pd.to_numeric(
        labelled["minimumExperience"], errors="coerce"
    ).astype("Int64")
    labelled["maximum_experience"] = pd.to_numeric(
        labelled["maximumExperience"], errors="coerce"
    ).astype("Int64")
    labelled["experience_band"] = labelled["minimum_experience"].map(experience_band)
    labelled["salary_disclosed"] = (
        labelled["salary"].map(normalized_key).ne("not disclosed")
        & labelled["salary"].notna()
    )
    for source_col, output_col in (
        ("minimumSalary", "minimum_salary"),
        ("maximumSalary", "maximum_salary"),
    ):
        values = pd.to_numeric(labelled[source_col], errors="coerce").astype("Int64")
        labelled[output_col] = values.mask(~labelled["salary_disclosed"])

    jobs = pd.DataFrame(
        {
            "job_id": labelled["jobId"].astype("int64"),
            "role_category": labelled["role_category"],
            "title_original": labelled["title"].map(normalized_text),
            "title_normalized": labelled["title_normalized"],
            "company_id": labelled["companyId"].astype("int64"),
            "company_name": labelled["company_name"],
            "company_key": labelled["company_key"],
            "location_original": labelled["location_original"],
            "primary_city": labelled["primary_city"],
            "work_arrangement": labelled["work_arrangement"],
            "location_count": labelled["location_count"].astype("int64"),
            "minimum_experience": labelled["minimum_experience"],
            "maximum_experience": labelled["maximum_experience"],
            "experience_band": labelled["experience_band"],
            "salary_original": labelled["salary"].map(normalized_text),
            "salary_disclosed": labelled["salary_disclosed"].astype(bool),
            "minimum_salary": labelled["minimum_salary"],
            "maximum_salary": labelled["maximum_salary"],
            "currency": labelled["currency"].map(normalized_text),
            "job_uploaded_relative": labelled["jobUploaded"].map(normalized_text),
            "reviews_count": pd.to_numeric(labelled["ReviewsCount"], errors="coerce").astype("Int64"),
            "aggregate_rating": pd.to_numeric(labelled["AggregateRating"], errors="coerce"),
            "description_clean": labelled["description_clean"],
            "template_group_id": labelled["template_group_id"],
        }
    ).sort_values("job_id", kind="mergesort").reset_index(drop=True)

    skill_records: list[dict[str, object]] = []
    jobs_without_skills = 0
    for row in labelled[["jobId", "tagsAndSkills"]].itertuples(index=False):
        if pd.isna(row.tagsAndSkills) or not normalized_text(row.tagsAndSkills):
            jobs_without_skills += 1
            continue
        seen: set[str] = set()
        for raw_skill in str(row.tagsAndSkills).split(","):
            skill_key, skill = canonical_skill(raw_skill, skill_aliases)
            if not skill_key or skill_key in seen:
                continue
            seen.add(skill_key)
            skill_records.append({"job_id": int(row.jobId), "skill_key": skill_key, "skill": skill})
    job_skills = pd.DataFrame(skill_records, columns=["job_id", "skill_key", "skill"])
    job_skills = job_skills.sort_values(["job_id", "skill_key"], kind="mergesort").reset_index(drop=True)

    source_descriptions = labelled["jobDescription"].fillna("").astype(str)
    template_sizes = jobs["template_group_id"].value_counts()
    repeated_template_sizes = template_sizes[template_sizes > 1]
    metrics: dict[str, object] = {
        "source_rows": len(source),
        "matched_postings": matched_count,
        "collision_postings_withheld": collision_count,
        "rule_labelled_postings": rule_labelled_count,
        "reviewed_near_misses_added": reviewed_near_misses_added,
        "reviewed_role_corrections_applied": reviewed_role_corrections,
        "reviewed_title_overrides_available": len(REVIEWED_TITLE_OVERRIDES),
        "reviewed_title_overrides_applied": reviewed_override_count,
        "unambiguous_labelled_postings": unambiguous_count,
        "duplicate_job_ids_removed": duplicate_job_ids_removed,
        "exact_cross_id_duplicates_removed": exact_cross_id_duplicates_removed,
        "retained_jobs": len(jobs),
        "job_skill_pairs": len(job_skills),
        "jobs_without_skills": jobs_without_skills,
        "descriptions_with_html_cleaned": int(source_descriptions.str.contains(HTML_TAG_PATTERN).sum()),
        "descriptions_with_email_redacted": int(source_descriptions.str.contains(EMAIL_PATTERN).sum()),
        "descriptions_with_phone_redacted": int(source_descriptions.str.contains(PHONE_PATTERN).sum()),
        "template_groups": int(len(template_sizes)),
        "repeated_template_groups": int(len(repeated_template_sizes)),
        "jobs_in_repeated_template_groups": int(repeated_template_sizes.sum()),
        "largest_template_group_size": int(template_sizes.max()),
        "role_counts": {role: int((jobs["role_category"] == role).sum()) for role in ROLE_LABELS},
    }
    return jobs, job_skills, metrics
