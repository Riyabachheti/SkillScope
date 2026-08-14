"""Deterministic sampling and independent review metrics for role labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from skillscope.taxonomy import ROLE_LABELS, match_title


REVIEW_DECISIONS = ("accept", "correct", "exclude")


class ReviewGateError(ValueError):
    """Raised when review artifacts are missing, stale, incomplete, or too weak."""


def stable_rank(seed: int, normalized_title: str) -> str:
    return hashlib.sha256(f"{seed}:{normalized_title}".encode("utf-8")).hexdigest()


def review_id(normalized_title: str) -> str:
    return hashlib.sha256(normalized_title.encode("utf-8")).hexdigest()[:16]


def classify_titles(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply taxonomy rules while retaining every collision and near miss."""
    required = {"jobId", "title", "companyName", "tagsAndSkills"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing review input columns: {sorted(missing)}")

    classified = frame.copy()
    matches = classified["title"].map(match_title)
    classified["normalized_title"] = matches.map(lambda item: item.normalized_title)
    classified["matched_roles"] = matches.map(lambda item: list(item.matched_roles))
    classified["predicted_role"] = matches.map(lambda item: item.predicted_role)
    classified["stratum"] = matches.map(lambda item: item.stratum)
    return classified


def unique_title_catalog(classified: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated postings so frequent titles cannot dominate review."""
    usable = classified[classified["normalized_title"].ne("")].copy()
    usable["title_frequency"] = usable.groupby("normalized_title")["jobId"].transform("size")
    catalog = (
        usable.sort_values(["normalized_title", "jobId"], kind="stable")
        .drop_duplicates("normalized_title", keep="first")
        .reset_index(drop=True)
    )
    return catalog


def build_review_sample(
    frame: pd.DataFrame,
    sample_per_role: int = 30,
    extra_sample_size: int = 30,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create stable role, collision, and near-miss strata from unique titles."""
    if sample_per_role <= 0 or extra_sample_size <= 0:
        raise ValueError("Sample sizes must be positive")

    catalog = unique_title_catalog(classify_titles(frame))
    catalog["stable_rank"] = catalog["normalized_title"].map(lambda title: stable_rank(seed, title))
    selected: list[pd.DataFrame] = []

    for role in ROLE_LABELS:
        role_rows = catalog[catalog["predicted_role"].eq(role)].sort_values("stable_rank").head(sample_per_role)
        if len(role_rows) < sample_per_role:
            raise ValueError(f"Only {len(role_rows)} unique titles available for {role}")
        selected.append(role_rows)

    ambiguous = catalog[catalog["stratum"].eq("ambiguous")].sort_values("stable_rank").head(extra_sample_size)
    near_miss = catalog[catalog["stratum"].eq("near_miss")].sort_values("stable_rank").head(extra_sample_size)
    selected.extend([ambiguous, near_miss])

    sample = pd.concat(selected, ignore_index=True)
    records: list[dict[str, Any]] = []
    for row in sample.itertuples(index=False):
        predicted_role = None if pd.isna(row.predicted_role) else str(row.predicted_role)
        records.append(
            {
                "review_id": review_id(row.normalized_title),
                "stratum": row.stratum,
                "title": str(row.title),
                "normalized_title": row.normalized_title,
                "title_frequency": int(row.title_frequency),
                "matched_roles": list(row.matched_roles),
                "predicted_role": predicted_role,
                "example_job_id": int(row.jobId),
                "example_company": None if pd.isna(row.companyName) else str(row.companyName),
                "example_skills": None if pd.isna(row.tagsAndSkills) else str(row.tagsAndSkills),
            }
        )
    return records


def validate_reviews(
    sample: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    minimum_precision: float = 0.90,
) -> dict[str, Any]:
    """Validate annotation completeness and calculate per-role precision."""
    review_by_id = {item["review_id"]: item for item in reviews}
    if len(review_by_id) != len(reviews):
        raise ValueError("Review labels contain duplicate review_id values")

    missing_ids = [item["review_id"] for item in sample if item["review_id"] not in review_by_id]
    unknown_ids = sorted(set(review_by_id).difference(item["review_id"] for item in sample))
    invalid_reviews: list[str] = []
    reviewed_rows: list[dict[str, Any]] = []

    for item in sample:
        review = review_by_id.get(item["review_id"])
        if review is None:
            continue
        decision = review.get("decision")
        reviewed_role = review.get("reviewed_role")
        if decision not in REVIEW_DECISIONS:
            invalid_reviews.append(item["review_id"])
            continue
        if decision in {"accept", "correct"} and reviewed_role not in ROLE_LABELS:
            invalid_reviews.append(item["review_id"])
            continue
        if decision == "exclude" and reviewed_role is not None:
            invalid_reviews.append(item["review_id"])
            continue
        reviewed_rows.append({**item, **review})

    per_role: dict[str, Any] = {}
    threshold_failures: list[str] = []
    for role in ROLE_LABELS:
        role_rows = [row for row in reviewed_rows if row["predicted_role"] == role]
        correct = sum(row["reviewed_role"] == role for row in role_rows)
        precision = correct / len(role_rows) if role_rows else 0.0
        per_role[role] = {
            "reviewed": len(role_rows),
            "correct": correct,
            "precision": round(precision, 4),
        }
        if precision < minimum_precision:
            threshold_failures.append(role)

    near_miss_rows = [row for row in reviewed_rows if row["stratum"] == "near_miss"]
    ambiguous_rows = [row for row in reviewed_rows if row["stratum"] == "ambiguous"]
    missed_in_scope = sum(row["reviewed_role"] in ROLE_LABELS for row in near_miss_rows)

    return {
        "complete": not missing_ids and not unknown_ids and not invalid_reviews,
        "passes_precision_threshold": not threshold_failures,
        "minimum_precision": minimum_precision,
        "sample_count": len(sample),
        "reviewed_count": len(reviewed_rows),
        "missing_review_ids": missing_ids,
        "unknown_review_ids": unknown_ids,
        "invalid_review_ids": invalid_reviews,
        "threshold_failures": threshold_failures,
        "per_role": per_role,
        "near_miss_reviewed": len(near_miss_rows),
        "near_miss_in_scope": missed_in_scope,
        "ambiguous_reviewed": len(ambiguous_rows),
    }


def materialize_review_labels(
    sample: list[dict[str, Any]], annotation: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expand compact, independently reviewed judgments into row-level labels."""
    reviewed_ids = annotation.get("reviewed_ids", [])
    if len(reviewed_ids) != len(set(reviewed_ids)):
        raise ValueError("Annotation contains duplicate reviewed_ids")
    sample_ids = [item["review_id"] for item in sample]
    if set(reviewed_ids) != set(sample_ids):
        raise ValueError("Annotation reviewed_ids do not match the generated sample")

    overrides = annotation.get("overrides", {})
    if not set(overrides).issubset(sample_ids):
        raise ValueError("Annotation overrides contain an unknown review_id")

    labels: list[dict[str, Any]] = []
    for item in sample:
        if item["predicted_role"]:
            label = {
                "decision": "accept",
                "reviewed_role": item["predicted_role"],
                "review_notes": "Reviewed title and example skills; candidate label accepted.",
            }
        else:
            label = {
                "decision": "exclude",
                "reviewed_role": None,
                "review_notes": "Reviewed as hybrid, ambiguous, or outside the four-role scope.",
            }
        label.update(overrides.get(item["review_id"], {}))
        labels.append({"review_id": item["review_id"], **label})
    return labels


def validate_review_gate(
    sample_path: Path,
    annotation_path: Path,
    expected_sample: list[dict[str, Any]] | None = None,
    minimum_precision: float = 0.90,
) -> dict[str, Any]:
    """Require review artifacts to match the current sample and quality threshold."""
    for path in (sample_path, annotation_path):
        if not path.is_file():
            raise ReviewGateError(f"Required review file not found: {path}")

    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if expected_sample is not None and sample != expected_sample:
        raise ReviewGateError(
            "Role-review sample is stale for the current source or taxonomy; "
            "regenerate and review it before building"
        )

    sample_sha256 = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    if annotation.get("sample_sha256") != sample_sha256:
        raise ReviewGateError("Annotation was created for a different review sample")

    try:
        reviews = materialize_review_labels(sample, annotation)
    except ValueError as error:
        raise ReviewGateError(str(error)) from error
    metrics = validate_reviews(sample, reviews, minimum_precision=minimum_precision)
    if not metrics["complete"]:
        raise ReviewGateError("Review labels are incomplete or invalid")
    if not metrics["passes_precision_threshold"]:
        raise ReviewGateError("At least one role is below the precision threshold")

    metrics["sample_sha256"] = sample_sha256
    metrics["review_policy"] = annotation.get("review_policy")
    return metrics


def write_json(payload: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
