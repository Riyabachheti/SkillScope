import unittest
from pathlib import Path

import duckdb
import pandas as pd

from skillscope.analysis import query_name, run_queries, validate_results


class AnalysisTests(unittest.TestCase):
    def test_query_name_removes_numeric_prefix(self):
        self.assertEqual(query_name(Path("03_top_skills_by_role.sql")), "top_skills_by_role")

    def test_all_queries_run_against_contract_shaped_tables(self):
        jobs = pd.DataFrame(
            {
                "job_id": [1, 2, 3, 4],
                "role_category": [
                    "Data Analyst / BI",
                    "Data Scientist",
                    "Data Engineer",
                    "ML / AI Engineer",
                ],
                "experience_band": [
                    "Entry level (0-2 years)",
                    "Early career (3-5 years)",
                    "Mid level (6-9 years)",
                    "Senior (10+ years)",
                ],
                "primary_city": ["Delhi", "Delhi", "Remote", "Pune"],
                "company_key": ["a", "a", "b", "c"],
                "company_name": ["A", "A", "B", "C"],
            }
        )
        skills = pd.DataFrame(
            {
                "job_id": [1, 1, 2, 2, 3, 3, 4, 4],
                "skill_key": ["python", "sql", "python", "ml", "python", "sql", "ml", "python"],
                "skill": ["Python", "SQL", "Python", "Machine Learning", "Python", "SQL", "Machine Learning", "Python"],
            }
        )
        connection = duckdb.connect(database=":memory:")
        connection.register("jobs", jobs)
        connection.register("job_skills", skills)
        results = run_queries(connection, Path(__file__).resolve().parents[1] / "sql")
        self.assertEqual(validate_results(results), [])
        self.assertEqual(results["role_distribution"]["posting_count"].sum(), 4)
        self.assertEqual(results["entry_level_skills"]["denominator"].iloc[0], 1)

    def test_validation_rejects_out_of_range_percentage(self):
        connection = duckdb.connect(database=":memory:")
        jobs = pd.DataFrame(
            {
                "job_id": [1, 2, 3, 4],
                "role_category": ["A", "B", "C", "D"],
                "experience_band": ["Entry level (0-2 years)"] * 4,
                "primary_city": ["Pune"] * 4,
                "company_key": ["a"] * 4,
                "company_name": ["A"] * 4,
            }
        )
        skills = pd.DataFrame(
            {"job_id": [1, 2, 3, 4], "skill_key": ["sql"] * 4, "skill": ["SQL"] * 4}
        )
        connection.register("jobs", jobs)
        connection.register("job_skills", skills)
        results = run_queries(connection, Path(__file__).resolve().parents[1] / "sql")
        results["role_distribution"].loc[0, "share_pct"] = 101.0
        self.assertTrue(any("between 0 and 100" in error for error in validate_results(results)))


if __name__ == "__main__":
    unittest.main()
