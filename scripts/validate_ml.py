#!/usr/bin/env python3
"""Validate saved Phase 6 predictions, metrics, grouping, and model reload."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skillscope.ml import (  # noqa: E402
    MODEL_NAMES,
    PREDICTION_COLUMNS,
    build_model_frame,
    metric_summary,
    predict_with_model,
    validate_experiment_result,
)
from skillscope.transform import file_sha256  # noqa: E402


def main() -> int:
    jobs_path = ROOT / "data" / "processed" / "jobs.parquet"
    skills_path = ROOT / "data" / "processed" / "job_skills.parquet"
    metrics_path = ROOT / "reports" / "generated" / "ml_metrics.json"
    predictions_path = ROOT / "reports" / "generated" / "ml_predictions.csv"
    model_path = ROOT / "models" / "role_classifier.joblib"
    transformation_metrics_path = (
        ROOT / "reports" / "generated" / "transformation_metrics.json"
    )
    required = (
        jobs_path,
        skills_path,
        transformation_metrics_path,
        metrics_path,
        predictions_path,
        model_path,
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        print(f"Missing Phase 6 inputs or artifacts: {missing}", file=sys.stderr)
        return 1

    try:
        import joblib

        jobs = pd.read_parquet(jobs_path)
        job_skills = pd.read_parquet(skills_path)
        frame = build_model_frame(jobs, job_skills)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(predictions_path)
        model_artifact = joblib.load(model_path)

        errors = validate_experiment_result(metrics, predictions, frame)
        actual_hashes = {
            "jobs_sha256": file_sha256(jobs_path),
            "job_skills_sha256": file_sha256(skills_path),
        }
        transformation_metrics = json.loads(
            transformation_metrics_path.read_text(encoding="utf-8")
        )
        expected_phase3_hashes = {
            "jobs_sha256": transformation_metrics.get("jobs_sha256"),
            "job_skills_sha256": transformation_metrics.get("job_skills_sha256"),
        }
        if actual_hashes != expected_phase3_hashes:
            errors.append("current Parquet hashes do not match Phase 3 metrics")
        if metrics.get("inputs") != actual_hashes:
            errors.append("ML metric input hashes do not match current Parquet files")
        if model_artifact.get("input_hashes") != actual_hashes:
            errors.append("serialized model input hashes do not match current Parquet files")
        if model_artifact.get("selected_model") != metrics.get("selected_model"):
            errors.append("serialized and metric selected-model names disagree")
        if model_artifact.get("excluded_leakage_columns") != metrics["experiment"].get(
            "excluded_leakage_columns"
        ):
            errors.append("serialized and metric leakage exclusions disagree")
        if model_artifact.get("feature_sources") != metrics["experiment"].get(
            "feature_sources"
        ):
            errors.append("serialized and metric feature sources disagree")

        for name in MODEL_NAMES:
            prediction_column = PREDICTION_COLUMNS[name]
            recomputed = metric_summary(
                predictions["actual_role"], predictions[prediction_column]
            )
            saved = metrics["models"][name]
            for key in ("accuracy", "macro_f1", "per_role", "confusion_matrix"):
                if recomputed[key] != saved[key]:
                    errors.append(f"{name}.{key} does not match saved predictions")
            fold_scores = []
            for fold_number in range(1, 6):
                fold_rows = predictions[predictions["fold"] == fold_number]
                recomputed_fold = metric_summary(
                    fold_rows["actual_role"], fold_rows[prediction_column]
                )
                saved_fold = next(
                    item for item in saved["folds"] if item["fold"] == fold_number
                )
                if saved_fold["test_rows"] != len(fold_rows):
                    errors.append(f"{name} fold {fold_number} row count is stale")
                for key in ("accuracy", "macro_f1", "per_role", "confusion_matrix"):
                    if recomputed_fold[key] != saved_fold[key]:
                        errors.append(f"{name} fold {fold_number}.{key} is stale")
                fold_scores.append(float(recomputed_fold["macro_f1"]))
            mean_score = round(statistics.fmean(fold_scores), 6)
            std_score = round(statistics.pstdev(fold_scores), 6)
            if saved["fold_macro_f1_mean"] != mean_score:
                errors.append(f"{name} fold macro F1 mean is stale")
            if saved["fold_macro_f1_std"] != std_score:
                errors.append(f"{name} fold macro F1 standard deviation is stale")

        smoke_test = model_artifact.get("smoke_test", [])
        if len(smoke_test) != 12:
            errors.append("serialized model smoke test must contain 12 rows")
        smoke_ids = [item["job_id"] for item in smoke_test]
        smoke_frame = frame.set_index("job_id").loc[smoke_ids].reset_index()
        smoke_predictions = predict_with_model(
            model_artifact["model_bundle"], smoke_frame
        )
        expected_smoke = [item["prediction"] for item in smoke_test]
        if smoke_predictions != expected_smoke:
            errors.append("reloaded final model failed its prediction smoke test")
    except (ImportError, KeyError, OSError, RuntimeError, StopIteration, ValueError) as error:
        print(f"Phase 6 validation error: {error}", file=sys.stderr)
        return 1

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "PASS: Phase 6 artifacts match current inputs; all five OOF predictions, "
        "company/template folds, metrics, selection, and model reload are valid."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
