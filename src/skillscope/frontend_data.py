"""Read-only artifact contract for the local SkillScope frontend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from skillscope.analysis import QUERY_FILES, query_name, validate_results
from skillscope.source import sha256_file
from skillscope.taxonomy import ROLE_LABELS


EXPECTED_SELECTED_MODEL = "combined_logistic_regression"
REQUIRED_MODEL_NAMES = {
    "majority_baseline",
    "skills_logistic_regression",
    "skills_linear_svm",
    "combined_logistic_regression",
    "combined_linear_svm",
}


class FrontendDataError(RuntimeError):
    """Raised when local presentation artifacts are missing, stale, or invalid."""


@dataclass(frozen=True)
class FrontendData:
    """Verified aggregate inputs used by the presentation layer."""

    results: dict[str, pd.DataFrame]
    ml_metrics: dict[str, Any]
    transformation_metrics: dict[str, Any]
    posting_count: int
    job_skill_pairs: int

    @property
    def selected_model(self) -> dict[str, Any]:
        return self.ml_metrics["models"][EXPECTED_SELECTED_MODEL]


def load_frontend_data(project_root: Path) -> FrontendData:
    """Load current Phase 4–6 aggregates and reject inconsistent artifacts."""
    root = Path(project_root).resolve()
    generated = root / "reports" / "generated"
    analysis_dir = generated / "analysis"
    jobs_path = root / "data" / "processed" / "jobs.parquet"
    skills_path = root / "data" / "processed" / "job_skills.parquet"
    summary_path = generated / "analysis_summary.json"
    ml_path = generated / "ml_metrics.json"
    transformation_path = generated / "transformation_metrics.json"

    required = [jobs_path, skills_path, summary_path, ml_path, transformation_path]
    required.extend(
        analysis_dir / f"{query_name(Path(filename))}.csv" for filename in QUERY_FILES
    )
    missing = [path.relative_to(root).as_posix() for path in required if not path.is_file()]
    if missing:
        raise FrontendDataError(
            "Missing local frontend inputs: "
            + ", ".join(missing)
            + ". Rebuild and validate Phases 3–6 before starting the app."
        )

    actual_inputs = {
        "jobs_sha256": sha256_file(jobs_path),
        "job_skills_sha256": sha256_file(skills_path),
    }
    summary = _read_json(summary_path)
    if summary.get("inputs") != actual_inputs:
        raise FrontendDataError(
            "Analysis artifacts are stale for the current processed tables. "
            "Run scripts/run_analysis.py and scripts/validate_analysis.py."
        )

    results: dict[str, pd.DataFrame] = {}
    for filename in QUERY_FILES:
        name = query_name(Path(filename))
        try:
            results[name] = pd.read_csv(analysis_dir / f"{name}.csv")
        except Exception as error:  # pandas provides detailed parsing causes
            raise FrontendDataError(f"Could not read {name}.csv: {error}") from error

    errors = validate_results(results)
    if errors:
        raise FrontendDataError("Invalid analysis artifacts: " + "; ".join(errors))
    _validate_summary_rows(results, summary)
    _validate_roles_and_ordering(results)

    transformation = _read_json(transformation_path)
    if {
        "jobs_sha256": transformation.get("jobs_sha256"),
        "job_skills_sha256": transformation.get("job_skills_sha256"),
    } != actual_inputs:
        raise FrontendDataError(
            "Transformation metrics do not match the current processed tables. "
            "Run scripts/build_processed_data.py and scripts/validate_processed_data.py."
        )

    posting_count = int(results["role_distribution"]["posting_count"].sum())
    job_skill_pairs = int(transformation.get("job_skill_pairs", -1))
    if posting_count != int(transformation.get("retained_jobs", -1)):
        raise FrontendDataError("Frontend posting total does not match transformation metrics")
    if job_skill_pairs < 1:
        raise FrontendDataError("Transformation metrics contain no job-skill pair total")

    ml_metrics = _read_json(ml_path)
    _validate_ml_metrics(ml_metrics, actual_inputs, posting_count)
    return FrontendData(
        results=results,
        ml_metrics=ml_metrics,
        transformation_metrics=transformation,
        posting_count=posting_count,
        job_skill_pairs=job_skill_pairs,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrontendDataError(f"Could not read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise FrontendDataError(f"{path.name} must contain a JSON object")
    return value


def _validate_summary_rows(
    results: dict[str, pd.DataFrame], summary: dict[str, Any]
) -> None:
    summary_results = summary.get("results")
    if not isinstance(summary_results, dict) or set(summary_results) != set(results):
        raise FrontendDataError("Analysis summary result set does not match the CSV files")
    for name, frame in results.items():
        expected = pd.DataFrame(summary_results[name], columns=frame.columns)
        try:
            pd.testing.assert_frame_equal(frame, expected, check_dtype=False)
        except AssertionError as error:
            raise FrontendDataError(
                f"{name}.csv does not match analysis_summary.json"
            ) from error


def _validate_roles_and_ordering(results: dict[str, pd.DataFrame]) -> None:
    expected_roles = set(ROLE_LABELS)
    actual_roles = set(results["role_distribution"]["role_category"])
    role_skill_roles = set(results["top_skills_by_role"]["role_category"])
    if actual_roles != expected_roles or role_skill_roles != expected_roles:
        raise FrontendDataError("Frontend artifacts do not contain the four reviewed roles")

    _require_rank_sequence(results["top_skills_overall"], "skill_rank")
    _require_rank_sequence(results["entry_level_skills"], "skill_rank")
    _require_rank_sequence(results["skill_cooccurrence"], "pair_rank")
    _require_rank_sequence(results["company_concentration"], "company_rank")
    for role, frame in results["top_skills_by_role"].groupby(
        "role_category", sort=False
    ):
        _require_rank_sequence(frame, "skill_rank", context=str(role))


def _require_rank_sequence(
    frame: pd.DataFrame, column: str, context: str | None = None
) -> None:
    actual = [int(value) for value in frame[column].tolist()]
    expected = list(range(1, len(frame) + 1))
    if actual != expected:
        suffix = f" for {context}" if context else ""
        raise FrontendDataError(f"{column} is not a stable 1-based sequence{suffix}")


def _validate_ml_metrics(
    metrics: dict[str, Any], actual_inputs: dict[str, str], posting_count: int
) -> None:
    if metrics.get("inputs") != actual_inputs:
        raise FrontendDataError(
            "ML metrics are stale for the current processed tables. "
            "Run scripts/run_ml_experiment.py and scripts/validate_ml.py."
        )
    if metrics.get("selected_model") != EXPECTED_SELECTED_MODEL:
        raise FrontendDataError("ML metrics do not contain the approved selected model")
    models = metrics.get("models")
    if not isinstance(models, dict) or set(models) != REQUIRED_MODEL_NAMES:
        raise FrontendDataError("ML metrics do not contain the five approved comparisons")
    experiment = metrics.get("experiment")
    if not isinstance(experiment, dict) or int(experiment.get("rows", -1)) != posting_count:
        raise FrontendDataError("ML row count does not match the frontend posting total")
    if experiment.get("template_groups_kept_intact") is not True:
        raise FrontendDataError("ML metrics do not confirm intact template groups")
    for name, model in models.items():
        if not isinstance(model, dict):
            raise FrontendDataError(f"Invalid ML metric object for {name}")
        for metric_name in ("macro_f1", "accuracy"):
            value = model.get(metric_name)
            if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                raise FrontendDataError(f"Invalid {metric_name} for {name}")
    selected = models[EXPECTED_SELECTED_MODEL]
    matrix = selected.get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != len(ROLE_LABELS)
        or any(not isinstance(row, list) or len(row) != len(ROLE_LABELS) for row in matrix)
    ):
        raise FrontendDataError("Selected-model confusion matrix must be 4 by 4")
