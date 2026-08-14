import unittest

import pandas as pd

from skillscope.contracts import validate_processed_tables, validate_transformation_counts


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.jobs = pd.DataFrame(
            {
                "job_id": [1],
                "role_category": ["Data Analyst / BI"],
                "title_original": ["Data Analyst"],
                "title_normalized": ["data analyst"],
                "company_id": [10],
                "company_name": ["Example Co"],
                "company_key": ["example co"],
                "location_original": ["Bengaluru"],
                "primary_city": ["Bengaluru"],
                "location_count": [1],
                "work_arrangement": ["Not specified"],
                "minimum_experience": [0],
                "maximum_experience": [2],
                "experience_band": ["Entry level (0-2 years)"],
                "salary_original": ["Not disclosed"],
                "salary_disclosed": [False],
                "minimum_salary": [pd.NA],
                "maximum_salary": [pd.NA],
                "currency": ["INR"],
                "job_uploaded_relative": ["Today"],
                "reviews_count": [5],
                "aggregate_rating": [4.0],
                "description_clean": ["Analyze data"],
                "template_group_id": ["a" * 64],
            }
        )
        self.skills = pd.DataFrame(
            {"job_id": [1], "skill_key": ["sql"], "skill": ["SQL"]}
        )

    def test_valid_tables_have_no_errors(self):
        self.assertEqual(validate_processed_tables(self.jobs, self.skills), [])

    def test_contract_reports_duplicate_ids_orphans_and_pii(self):
        jobs = pd.concat([self.jobs, self.jobs], ignore_index=True)
        jobs.loc[0, "description_clean"] = "jobs@example.com"
        skills = pd.DataFrame(
            {"job_id": [99], "skill_key": ["sql"], "skill": ["SQL"]}
        )
        errors = validate_processed_tables(jobs, skills)
        self.assertTrue(any("job_id" in error for error in errors))
        self.assertTrue(any("email" in error for error in errors))
        self.assertTrue(any("orphan" in error for error in errors))

    def test_contract_rejects_invalid_template_group_id(self):
        jobs = self.jobs.copy()
        jobs.loc[0, "template_group_id"] = "not-a-hash"
        errors = validate_processed_tables(jobs, self.skills)
        self.assertTrue(any("template_group_id" in error for error in errors))

    def test_contract_reports_missing_columns_without_crashing(self):
        jobs = self.jobs.drop(columns=["role_category", "company_key"])
        skills = self.skills.drop(columns=["skill"])
        errors = validate_processed_tables(jobs, skills)
        self.assertTrue(any("jobs is missing required columns" in error for error in errors))
        self.assertTrue(any("job_skills is missing required columns" in error for error in errors))

    def test_transformation_count_stages_reconcile(self):
        metrics = self._valid_metrics()
        self.assertEqual(
            validate_transformation_counts(metrics, self.jobs, self.skills),
            [],
        )

    def test_transformation_count_contract_detects_broken_stage(self):
        metrics = self._valid_metrics()
        metrics["retained_jobs"] = 2
        errors = validate_transformation_counts(metrics, self.jobs, self.skills)
        self.assertTrue(any("retained count" in error for error in errors))
        self.assertTrue(any("jobs table row count" in error for error in errors))

    def test_transformation_count_contract_requires_all_metrics(self):
        errors = validate_transformation_counts({}, self.jobs, self.skills)
        self.assertTrue(any("missing required keys" in error for error in errors))

    @staticmethod
    def _valid_metrics():
        return {
            "source_rows": 1,
            "matched_postings": 1,
            "collision_postings_withheld": 0,
            "rule_labelled_postings": 1,
            "reviewed_near_misses_added": 0,
            "reviewed_role_corrections_applied": 0,
            "reviewed_title_overrides_applied": 0,
            "unambiguous_labelled_postings": 1,
            "duplicate_job_ids_removed": 0,
            "exact_cross_id_duplicates_removed": 0,
            "retained_jobs": 1,
            "job_skill_pairs": 1,
            "jobs_without_skills": 0,
            "template_groups": 1,
            "repeated_template_groups": 0,
            "jobs_in_repeated_template_groups": 0,
            "largest_template_group_size": 1,
            "role_counts": {
                "Data Analyst / BI": 1,
                "Data Scientist": 0,
                "Data Engineer": 0,
                "ML / AI Engineer": 0,
            },
        }


if __name__ == "__main__":
    unittest.main()
