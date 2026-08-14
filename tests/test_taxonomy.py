from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from skillscope.taxonomy import (
    DATA_ANALYST_BI,
    DATA_ENGINEER,
    DATA_SCIENTIST,
    ML_AI_ENGINEER,
    match_title,
    match_title_with_review,
    normalize_title,
)


class TaxonomyUnitTests(unittest.TestCase):
    def test_normalize_title_standardizes_separators(self) -> None:
        self.assertEqual(normalize_title("  AI / ML—Engineer  "), "ai ml engineer")

    def test_expected_role_variants(self) -> None:
        cases = {
            "Senior Data Analyst": DATA_ANALYST_BI,
            "Business Intelligence Analyst": DATA_ANALYST_BI,
            "Lead Data Scientist - Forecasting": DATA_SCIENTIST,
            "Azure Data Engineer": DATA_ENGINEER,
            "ETL Developer": DATA_ENGINEER,
            "AI / ML Engineer": ML_AI_ENGINEER,
            "Machine Learning Engineer II": ML_AI_ENGINEER,
        }
        for title, expected in cases.items():
            with self.subTest(title=title):
                result = match_title(title)
                self.assertEqual(result.predicted_role, expected)
                self.assertEqual(result.stratum, "candidate")

    def test_collision_is_never_silently_assigned(self) -> None:
        result = match_title("Data Scientist and ML Engineer")
        self.assertIsNone(result.predicted_role)
        self.assertEqual(result.stratum, "ambiguous")
        self.assertEqual(len(result.matched_roles), 2)
        reviewed_result = match_title_with_review("Data Scientist and ML Engineer")
        self.assertIsNone(reviewed_result.predicted_role)
        self.assertEqual(reviewed_result.stratum, "ambiguous")

    def test_near_miss_is_retained_for_review(self) -> None:
        result = match_title("Data Architect")
        self.assertIsNone(result.predicted_role)
        self.assertEqual(result.stratum, "near_miss")

    def test_review_driven_false_positives_are_excluded(self) -> None:
        for title in (
            "Accounting & Reporting Analyst-Payable",
            "Capital Reporting Analyst",
            "Regulatory Reporting Analyst",
        ):
            with self.subTest(title=title):
                self.assertNotEqual(match_title(title).predicted_role, DATA_ANALYST_BI)
        self.assertNotEqual(match_title("Data Scientist + Instructor").predicted_role, DATA_SCIENTIST)

    def test_review_driven_misses_are_included(self) -> None:
        cases = {
            "Retention Analytics Intern": DATA_ANALYST_BI,
            "Deputy National Lead - Data Science": DATA_SCIENTIST,
            "AI Solutions Engineer": ML_AI_ENGINEER,
            "Advisory Data Science Engineer": ML_AI_ENGINEER,
            "BI Engineer - Tableau": DATA_ANALYST_BI,
            "Analytics Technical Specialist": DATA_ANALYST_BI,
            "S&C GN - Data&AI - Resources - Analyst": DATA_ANALYST_BI,
        }
        for title, expected in cases.items():
            with self.subTest(title=title):
                self.assertEqual(match_title(title).predicted_role, expected)

    def test_six_exact_reviewed_title_overrides_are_applied(self) -> None:
        cases = {
            "AI AND DATA Scientist Engineer": ML_AI_ENGINEER,
            "SFDC Data Cloud Developer": DATA_ENGINEER,
            "VIS & BI Strategy Practitioner": DATA_ANALYST_BI,
            "Python Machine learning-PAN India": ML_AI_ENGINEER,
            "Sr. AI ML (GEN AI) - Agentic AI Professional": ML_AI_ENGINEER,
            "BI Reports Developer (Senior)": DATA_ANALYST_BI,
        }
        for title, expected in cases.items():
            with self.subTest(title=title):
                result = match_title_with_review(title)
                self.assertEqual(result.predicted_role, expected)
                self.assertEqual(result.stratum, "reviewed_override")

    def test_review_override_is_exact_not_a_broad_pattern(self) -> None:
        result = match_title_with_review("SFDC Data Cloud Developer Manager")
        self.assertIsNone(result.predicted_role)
        self.assertEqual(result.stratum, "near_miss")

    def test_unrelated_title_is_unmatched(self) -> None:
        result = match_title("Fire and Safety Officer")
        self.assertEqual(result.stratum, "unmatched")


if __name__ == "__main__":
    unittest.main()
