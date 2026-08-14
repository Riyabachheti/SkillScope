from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from skillscope.review import (
    ReviewGateError,
    build_review_sample,
    materialize_review_labels,
    validate_review_gate,
    validate_reviews,
    write_json,
)
from skillscope.taxonomy import ROLE_LABELS


def sample_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    templates = {
        "Data Analyst": "Data Analyst {index}",
        "Data Scientist": "Data Scientist {index}",
        "Data Engineer": "Data Engineer {index}",
        "ML Engineer": "ML Engineer {index}",
    }
    job_id = 1
    for template in templates.values():
        for index in range(4):
            rows.append(
                {
                    "jobId": job_id,
                    "title": template.format(index=index),
                    "companyName": f"Company {index}",
                    "tagsAndSkills": "Python,SQL",
                }
            )
            job_id += 1
    rows.extend(
        [
            {
                "jobId": job_id,
                "title": "Data Scientist and ML Engineer",
                "companyName": "Hybrid Co",
                "tagsAndSkills": "Python",
            },
            {
                "jobId": job_id + 1,
                "title": "Data Architect",
                "companyName": "Architecture Co",
                "tagsAndSkills": "SQL",
            },
        ]
    )
    return pd.DataFrame(rows)


class ReviewUnitTests(unittest.TestCase):
    def test_review_sample_is_deterministic_and_stratified(self) -> None:
        first = build_review_sample(sample_frame(), sample_per_role=2, extra_sample_size=1, seed=7)
        second = build_review_sample(sample_frame(), sample_per_role=2, extra_sample_size=1, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        for role in ROLE_LABELS:
            self.assertEqual(sum(item["predicted_role"] == role for item in first), 2)
        self.assertEqual(sum(item["stratum"] == "ambiguous" for item in first), 1)
        self.assertEqual(sum(item["stratum"] == "near_miss" for item in first), 1)
        self.assertTrue(
            all(
                item["predicted_role"] is None
                for item in first
                if item["stratum"] in {"ambiguous", "near_miss"}
            )
        )

    def test_validation_reports_complete_perfect_review(self) -> None:
        sample = build_review_sample(sample_frame(), sample_per_role=2, extra_sample_size=1, seed=7)
        reviews = []
        for item in sample:
            if item["predicted_role"]:
                reviews.append(
                    {
                        "review_id": item["review_id"],
                        "decision": "accept",
                        "reviewed_role": item["predicted_role"],
                        "review_notes": "fixture",
                    }
                )
            else:
                reviews.append(
                    {
                        "review_id": item["review_id"],
                        "decision": "exclude",
                        "reviewed_role": None,
                        "review_notes": "fixture",
                    }
                )
        metrics = validate_reviews(sample, reviews)
        self.assertTrue(metrics["complete"])
        self.assertTrue(metrics["passes_precision_threshold"])

    def test_validation_rejects_incomplete_review(self) -> None:
        sample = build_review_sample(sample_frame(), sample_per_role=2, extra_sample_size=1, seed=7)
        metrics = validate_reviews(sample, [])
        self.assertFalse(metrics["complete"])
        self.assertEqual(len(metrics["missing_review_ids"]), len(sample))

    def test_compact_annotations_require_every_review_id(self) -> None:
        sample = build_review_sample(sample_frame(), sample_per_role=2, extra_sample_size=1, seed=7)
        annotation = {"reviewed_ids": [item["review_id"] for item in sample], "overrides": {}}
        labels = materialize_review_labels(sample, annotation)
        self.assertEqual(len(labels), len(sample))
        with self.assertRaisesRegex(ValueError, "do not match"):
            materialize_review_labels(sample, {"reviewed_ids": [], "overrides": {}})

    def test_review_gate_accepts_current_complete_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path, annotation_path, sample = self._write_gate_files(Path(temp_dir))
            metrics = validate_review_gate(
                sample_path,
                annotation_path,
                expected_sample=sample,
            )
            self.assertTrue(metrics["complete"])
            self.assertTrue(metrics["passes_precision_threshold"])

    def test_review_gate_rejects_stale_sample_or_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path, annotation_path, sample = self._write_gate_files(Path(temp_dir))
            with self.assertRaisesRegex(ReviewGateError, "sample is stale"):
                validate_review_gate(
                    sample_path,
                    annotation_path,
                    expected_sample=[*sample, {"review_id": "new"}],
                )

            annotation = {
                "sample_sha256": "0" * 64,
                "reviewed_ids": [item["review_id"] for item in sample],
                "overrides": {},
            }
            write_json(annotation, annotation_path)
            with self.assertRaisesRegex(ReviewGateError, "different review sample"):
                validate_review_gate(sample_path, annotation_path)

    @staticmethod
    def _write_gate_files(directory: Path):
        sample = build_review_sample(
            sample_frame(), sample_per_role=2, extra_sample_size=1, seed=7
        )
        sample_path = directory / "sample.json"
        annotation_path = directory / "annotations.json"
        write_json(sample, sample_path)
        annotation = {
            "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            "reviewed_ids": [item["review_id"] for item in sample],
            "overrides": {},
        }
        write_json(annotation, annotation_path)
        return sample_path, annotation_path, sample


if __name__ == "__main__":
    unittest.main()
