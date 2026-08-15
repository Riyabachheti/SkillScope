#!/usr/bin/env python3
"""Run Phase 6 grouped evaluation and atomically write local artifacts."""

from __future__ import annotations

import json
import platform
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skillscope.ml import (  # noqa: E402
    EXCLUDED_LEAKAGE_COLUMNS,
    build_model_frame,
    predict_with_model,
    run_grouped_experiment,
)
from skillscope.transform import file_sha256  # noqa: E402


def checked_input_hashes(jobs_path: Path, skills_path: Path) -> dict[str, str]:
    """Require current Parquet hashes to match Phase 3 transformation metrics."""
    metrics_path = ROOT / "reports" / "generated" / "transformation_metrics.json"
    if not metrics_path.is_file():
        raise ValueError("transformation_metrics.json is missing; rebuild Phase 3 first")
    transformation_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    actual = {
        "jobs_sha256": file_sha256(jobs_path),
        "job_skills_sha256": file_sha256(skills_path),
    }
    for name, checksum in actual.items():
        if transformation_metrics.get(name) != checksum:
            raise ValueError(f"{name} does not match transformation_metrics.json")
    return actual


def main() -> int:
    jobs_path = ROOT / "data" / "processed" / "jobs.parquet"
    skills_path = ROOT / "data" / "processed" / "job_skills.parquet"
    if not jobs_path.is_file() or not skills_path.is_file():
        print(
            "Processed Parquet files are missing; run build_processed_data.py first.",
            file=sys.stderr,
        )
        return 1

    try:
        import joblib
        import sklearn

        input_hashes = checked_input_hashes(jobs_path, skills_path)
        jobs = pd.read_parquet(jobs_path)
        job_skills = pd.read_parquet(skills_path)
        metrics, predictions, model_bundle = run_grouped_experiment(jobs, job_skills)
        versions = {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        }
        metrics["inputs"] = input_hashes
        metrics["versions"] = versions

        model_frame = build_model_frame(jobs, job_skills)
        smoke_frame = model_frame.head(12)
        smoke_predictions = predict_with_model(model_bundle, smoke_frame)
        model_artifact = {
            "artifact_version": 1,
            "model_bundle": model_bundle,
            "selected_model": metrics["selected_model"],
            "feature_sources": metrics["experiment"]["feature_sources"],
            "excluded_leakage_columns": list(EXCLUDED_LEAKAGE_COLUMNS),
            "input_hashes": input_hashes,
            "versions": versions,
            "smoke_test": [
                {"job_id": int(job_id), "prediction": prediction}
                for job_id, prediction in zip(
                    smoke_frame["job_id"], smoke_predictions, strict=True
                )
            ],
        }
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"Phase 6 error: {error}", file=sys.stderr)
        return 1

    report_dir = ROOT / "reports" / "generated"
    model_dir = ROOT / "models"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "ml_metrics.json"
    predictions_path = report_dir / "ml_predictions.csv"
    model_path = model_dir / "role_classifier.joblib"

    try:
        with tempfile.TemporaryDirectory(prefix="phase6-", dir=ROOT) as temporary:
            temporary_dir = Path(temporary)
            temporary_metrics = temporary_dir / metrics_path.name
            temporary_predictions = temporary_dir / predictions_path.name
            temporary_model = temporary_dir / model_path.name
            temporary_metrics.write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            predictions.to_csv(temporary_predictions, index=False, lineterminator="\n")
            joblib.dump(model_artifact, temporary_model)

            loaded_artifact = joblib.load(temporary_model)
            loaded_predictions = predict_with_model(
                loaded_artifact["model_bundle"], smoke_frame
            )
            if loaded_predictions != smoke_predictions:
                raise ValueError("serialized model failed the prediction reload check")

            temporary_metrics.replace(metrics_path)
            temporary_predictions.replace(predictions_path)
            temporary_model.replace(model_path)
    except (OSError, ValueError) as error:
        print(f"Phase 6 artifact error: {error}", file=sys.stderr)
        return 1

    selected = str(metrics["selected_model"])
    macro_f1 = float(metrics["models"][selected]["macro_f1"])
    print(
        f"PASS: {selected} selected with company-grouped OOF macro F1 "
        f"{macro_f1:.3f}; wrote metrics, predictions, and model artifacts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
