import unittest

import pandas as pd

from skillscope.transform import (
    build_processed_tables,
    canonical_skill,
    clean_description,
    experience_band,
    make_template_group_id,
    parse_location,
)


class TransformTests(unittest.TestCase):
    def test_clean_description_removes_html_and_redacts_contact_details(self):
        raw = "<p>Email jobs@example.com</p><br>Call +91 98765-43210"
        actual = clean_description(raw)
        self.assertEqual(
            actual,
            "Email [EMAIL_REDACTED] Call [PHONE_REDACTED]",
        )

    def test_parse_location_preserves_multi_location_signal(self):
        parsed = parse_location(
            "Hybrid - Gurgaon/Gurugram, Bengaluru(Whitefield)",
            {"gurgaon/gurugram": "Gurugram"},
        )
        self.assertEqual(parsed.primary_city, "Gurugram")
        self.assertEqual(parsed.work_arrangement, "Hybrid")
        self.assertEqual(parsed.location_count, 2)

    def test_experience_band_uses_minimum_experience(self):
        self.assertEqual(experience_band(0), "Entry level (0-2 years)")
        self.assertEqual(experience_band(5), "Early career (3-5 years)")
        self.assertEqual(experience_band(8), "Mid level (6-9 years)")
        self.assertEqual(experience_band(10), "Senior (10+ years)")
        self.assertEqual(experience_band(None), "Unknown")

    def test_canonical_skill_merges_case_and_alias_variants(self):
        aliases = {
            "bi": "Business Intelligence",
            "powerbi": "Power BI",
            "power bi": "Power BI",
        }
        self.assertEqual(canonical_skill("PowerBI", aliases), ("power bi", "Power BI"))
        self.assertEqual(canonical_skill(" power bi ", aliases), ("power bi", "Power BI"))
        self.assertEqual(
            canonical_skill("BI", aliases),
            ("business intelligence", "Business Intelligence"),
        )

    def test_template_group_ignores_skill_and_experience_fields(self):
        first = make_template_group_id("Example Co", "<p>Build reports</p>")
        second = make_template_group_id(" example co ", "Build   reports")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_build_processed_tables_withholds_collisions_and_deduplicates(self):
        rows = [
            self._row(1, "Data Analyst", "SQL, sql, PowerBI"),
            self._row(1, "Data Analyst", "SQL, sql, PowerBI"),
            self._row(2, "Data Analyst", "Python, Power BI"),
            self._row(3, "Data Scientist and Data Engineer", "Python"),
            self._row(4, "Accountant", "Excel"),
            self._row(5, "Data Analyst", "SQL, sql, PowerBI"),
        ]
        # Same template, but a different skill and experience requirement: keep it.
        rows[2]["minimumExperience"] = 3
        rows[2]["maximumExperience"] = 5
        rows[2]["experience"] = "3-5 Yrs"
        # Same source values as job 1 except for jobId: remove it as an exact copy.
        rows[5] = {**rows[0], "jobId": 5}
        source = pd.DataFrame(rows)
        jobs, job_skills, metrics = build_processed_tables(
            source,
            {"bengaluru": "Bengaluru"},
            {"sql": "SQL", "powerbi": "Power BI", "power bi": "Power BI"},
        )
        self.assertEqual(len(jobs), 2)
        self.assertEqual(set(jobs["job_id"]), {1, 2})
        self.assertEqual(len(job_skills), 4)
        self.assertEqual(metrics["matched_postings"], 5)
        self.assertEqual(metrics["collision_postings_withheld"], 1)
        self.assertEqual(metrics["duplicate_job_ids_removed"], 1)
        self.assertEqual(metrics["exact_cross_id_duplicates_removed"], 1)
        self.assertEqual(metrics["repeated_template_groups"], 1)
        self.assertEqual(metrics["jobs_in_repeated_template_groups"], 2)
        self.assertEqual(jobs["template_group_id"].nunique(), 1)
        self.assertEqual(
            jobs.set_index("job_id").loc[2, "minimum_experience"],
            3,
        )

    def test_build_processed_tables_applies_all_six_reviewed_titles(self):
        reviewed_titles = {
            "AI AND DATA Scientist Engineer": "ML / AI Engineer",
            "SFDC Data Cloud Developer": "Data Engineer",
            "VIS & BI Strategy Practitioner": "Data Analyst / BI",
            "Python Machine learning-PAN India": "ML / AI Engineer",
            "Sr. AI ML (GEN AI) - Agentic AI Professional": "ML / AI Engineer",
            "BI Reports Developer (Senior)": "Data Analyst / BI",
        }
        rows = [
            self._row(index, title, "Python, SQL")
            for index, title in enumerate(reviewed_titles, start=10)
        ]
        jobs, _, metrics = build_processed_tables(
            pd.DataFrame(rows),
            {"bengaluru": "Bengaluru"},
            {"python": "Python", "sql": "SQL"},
        )
        actual = dict(zip(jobs["title_original"], jobs["role_category"]))
        self.assertEqual(actual, reviewed_titles)
        self.assertEqual(metrics["reviewed_title_overrides_available"], 6)
        self.assertEqual(metrics["reviewed_title_overrides_applied"], 6)

    @staticmethod
    def _row(job_id, title, skills):
        return {
            "title": title,
            "jobId": job_id,
            "currency": "INR",
            "jobUploaded": "Today",
            "companyName": "Example Co",
            "tagsAndSkills": skills,
            "experience": "0-2 Yrs",
            "salary": "Not disclosed",
            "location": "Bengaluru",
            "companyId": 10,
            "ReviewsCount": 5,
            "AggregateRating": 4.0,
            "jobDescription": "<p>Build reports</p>",
            "minimumSalary": 0,
            "maximumSalary": 0,
            "minimumExperience": 0,
            "maximumExperience": 2,
        }


if __name__ == "__main__":
    unittest.main()
