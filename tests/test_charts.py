from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd

from skillscope.charts import EXPECTED_CHARTS, create_charts, load_verified_results


def chart_results() -> dict[str, pd.DataFrame]:
    roles = ["Data Analyst / BI", "Data Scientist", "Data Engineer", "ML / AI Engineer"]
    return {
        "role_distribution": pd.DataFrame(
            {
                "role_category": roles,
                "posting_count": [10, 20, 40, 30],
                "share_pct": [10.0, 20.0, 40.0, 30.0],
            }
        ),
        "top_skills_overall": pd.DataFrame(
            {
                "skill_rank": [1],
                "skill": ["Python"],
                "posting_count": [60],
                "denominator": [100],
                "penetration_pct": [60.0],
            }
        ),
        "top_skills_by_role": pd.DataFrame(
            [
                {
                    "role_category": role,
                    "skill_rank": rank,
                    "skill": skill,
                    "posting_count": count,
                    "denominator": 100,
                    "penetration_pct": float(count),
                }
                for role in roles
                for rank, (skill, count) in enumerate(
                    [("Python", 60), ("SQL", 40), ("Cloud", 30), ("ETL", 20), ("BI", 10)],
                    start=1,
                )
            ]
        ),
        "entry_level_skills": pd.DataFrame(
            {
                "skill_rank": [1, 2],
                "skill": ["Python", "SQL"],
                "posting_count": [60, 40],
                "denominator": [100, 100],
                "penetration_pct": [60.0, 40.0],
            }
        ),
        "city_role_distribution": pd.DataFrame(
            [
                {
                    "primary_city": city,
                    "city_postings": 100,
                    "role_category": role,
                    "role_postings": 25,
                    "within_city_pct": 25.0,
                }
                for city in ("Bengaluru", "Hyderabad")
                for role in roles
            ]
        ),
        "skill_cooccurrence": pd.DataFrame(
            {
                "pair_rank": [1, 2],
                "skill_a": ["Python", "Python"],
                "skill_b": ["SQL", "Machine Learning"],
                "pair_posting_count": [30, 20],
                "skill_a_postings": [60, 60],
                "skill_b_postings": [40, 30],
                "denominator": [100, 100],
                "pair_penetration_pct": [30.0, 20.0],
                "lift": [1.25, 1.11],
            }
        ),
        "company_concentration": pd.DataFrame(
            {
                "company_rank": [1],
                "company_name": ["Example Co"],
                "posting_count": [10],
                "represented_roles": [4],
                "denominator": [100],
                "share_pct": [10.0],
            }
        ),
    }


class ChartTests(unittest.TestCase):
    def test_create_charts_writes_complete_png_and_svg_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = create_charts(chart_results(), Path(temp_dir))
            self.assertEqual(tuple(outputs), EXPECTED_CHARTS)
            self.assertEqual(sum(len(paths) for paths in outputs.values()), 10)
            for paths in outputs.values():
                png_path, svg_path = paths
                self.assertGreater(png_path.stat().st_size, 1_000)
                self.assertGreater(svg_path.stat().st_size, 1_000)
                self.assertEqual(png_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertIn("<svg", svg_path.read_text(encoding="utf-8")[:1_000])
                image = mpimg.imread(png_path)
                self.assertGreaterEqual(image.shape[0], 500)
                self.assertGreaterEqual(image.shape[1], 800)

    def test_chart_loader_rejects_stale_analysis_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jobs = root / "jobs.parquet"
            skills = root / "skills.parquet"
            jobs.write_bytes(b"jobs")
            skills.write_bytes(b"skills")
            summary = root / "analysis_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "inputs": {
                            "jobs_sha256": "0" * 64,
                            "job_skills_sha256": "0" * 64,
                        },
                        "results": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "stale"):
                load_verified_results(root, summary, jobs, skills)


if __name__ == "__main__":
    unittest.main()
