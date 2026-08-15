import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from skillscope.ml import (
    EXCLUDED_LEAKAGE_COLUMNS,
    MODEL_NAMES,
    ROLE_PROXY_PATTERN,
    build_model_frame,
    fit_feature_transformers,
    make_company_grouped_folds,
    predict_with_model,
    run_grouped_experiment,
    sanitize_description,
    select_model,
    transform_features,
)
from skillscope.taxonomy import ROLE_LABELS


SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


class MlFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        role_details = {
            "Data Analyst / BI": ("data analyst", "sql"),
            "Data Scientist": ("data scientist", "statistics"),
            "Data Engineer": ("data engineer", "pyspark"),
            "ML / AI Engineer": ("machine learning engineer", "tensorflow"),
        }
        rows = []
        skills = []
        for company_number in range(15):
            for role_number, role in enumerate(ROLE_LABELS):
                job_id = company_number * 10 + role_number
                role_phrase, skill = role_details[role]
                rows.append(
                    {
                        "job_id": job_id,
                        "role_category": role,
                        "company_key": f"company_{company_number}",
                        "template_group_id": f"template_{company_number}_{role_number}",
                        "description_clean": f"Work as a {role_phrase} building systems with Python",
                        "title_original": "DO NOT USE THIS TITLE",
                        "title_normalized": "do not use this title",
                    }
                )
                if job_id != 0:
                    skills.extend(
                        [
                            {"job_id": job_id, "skill_key": skill},
                            {"job_id": job_id, "skill_key": "python"},
                        ]
                    )
        self.jobs = pd.DataFrame(rows)
        self.skills = pd.DataFrame(skills)

    def test_model_frame_excludes_titles_and_keeps_zero_skill_rows(self) -> None:
        frame = build_model_frame(self.jobs, self.skills)
        self.assertEqual(len(frame), len(self.jobs))
        self.assertEqual(int(frame["skills"].map(len).eq(0).sum()), 1)
        self.assertTrue(
            {"title_original", "title_normalized"}.isdisjoint(frame.columns)
        )
        self.assertIn("job_id", EXCLUDED_LEAKAGE_COLUMNS)
        self.assertEqual(
            {"skills", "description_model"},
            {column for column in frame.columns if column in {"skills", "description_model"}},
        )
        self.assertFalse(frame["description_model"].str.contains("DO NOT USE", case=False).any())

    def test_description_sanitization_removes_compound_proxies_only(self) -> None:
        original = "Data Engineer partners with a platform engineer and ML engineers."
        sanitized = sanitize_description(original)
        self.assertIsNone(ROLE_PROXY_PATTERN.search(sanitized))
        self.assertIn("platform engineer", sanitized)

    def test_model_frame_does_not_modify_source_descriptions(self) -> None:
        before = self.jobs["description_clean"].copy(deep=True)
        frame = build_model_frame(self.jobs, self.skills)
        pd.testing.assert_series_equal(self.jobs["description_clean"], before)
        self.assertTrue(frame["role_proxy_matches"].gt(0).all())
        self.assertFalse(frame["description_model"].map(ROLE_PROXY_PATTERN.search).any())

    def test_model_frame_requires_company_groups(self) -> None:
        jobs = self.jobs.copy()
        jobs.loc[0, "company_key"] = ""
        with self.assertRaisesRegex(ValueError, "company_key"):
            build_model_frame(jobs, self.skills)

    def test_model_frame_rejects_duplicate_skills(self) -> None:
        duplicated = pd.concat([self.skills, self.skills.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate job_id/skill_key"):
            build_model_frame(self.jobs, duplicated)

    def test_selection_prefers_simpler_model_on_exact_tie(self) -> None:
        metrics = {
            "skills_logistic_regression": {"macro_f1": 0.5},
            "skills_linear_svm": {"macro_f1": 0.5},
            "combined_logistic_regression": {"macro_f1": 0.5},
            "combined_linear_svm": {"macro_f1": 0.5},
        }
        self.assertEqual(select_model(metrics), "skills_logistic_regression")

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
    def test_unknown_test_skill_does_not_enter_training_vocabulary(self) -> None:
        train = pd.DataFrame(
            {"skills": [("python",), ("sql",)], "description_model": ["one", "two"]}
        )
        test = pd.DataFrame(
            {"skills": [("test-only-skill",)], "description_model": ["three"]}
        )
        transformers = fit_feature_transformers(train, "skills")
        transformed = transform_features(test, transformers)
        self.assertEqual(set(transformers["skill_vectorizer"].feature_names_), {"python", "sql"})
        self.assertEqual(transformed.nnz, 0)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
    def test_company_folds_keep_companies_and_templates_apart(self) -> None:
        frame = build_model_frame(self.jobs, self.skills)
        folds = make_company_grouped_folds(frame)
        self.assertEqual(len(folds), 5)
        for train_index, test_index in folds:
            train = frame.iloc[train_index]
            test = frame.iloc[test_index]
            self.assertTrue(set(train["group_company"]).isdisjoint(test["group_company"]))
            self.assertTrue(set(train["group_template"]).isdisjoint(test["group_template"]))
            self.assertEqual(set(train["target"]), set(ROLE_LABELS))
            self.assertEqual(set(test["target"]), set(ROLE_LABELS))

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
    def test_grouped_experiment_is_complete_and_deterministic(self) -> None:
        first_metrics, first_predictions, _ = run_grouped_experiment(self.jobs, self.skills)
        second_metrics, second_predictions, _ = run_grouped_experiment(self.jobs, self.skills)
        self.assertEqual(first_metrics, second_metrics)
        pd.testing.assert_frame_equal(first_predictions, second_predictions)
        self.assertEqual(set(first_metrics["models"]), set(MODEL_NAMES))
        self.assertEqual(len(first_predictions), len(self.jobs))
        self.assertEqual(set(first_predictions["fold"]), {1, 2, 3, 4, 5})

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
    def test_model_bundle_round_trip_reproduces_predictions(self) -> None:
        import joblib

        _, _, model_bundle = run_grouped_experiment(self.jobs, self.skills)
        frame = build_model_frame(self.jobs, self.skills).head(8)
        expected = predict_with_model(model_bundle, frame)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(model_bundle, path)
            loaded = joblib.load(path)
        self.assertEqual(predict_with_model(loaded, frame), expected)


if __name__ == "__main__":
    unittest.main()
