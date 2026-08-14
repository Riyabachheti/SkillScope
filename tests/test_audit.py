from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from skillscope.audit import EXPECTED_COLUMNS, column_profile, normalize_text, sha256_file


class AuditUnitTests(unittest.TestCase):
    def test_normalize_text_trims_and_collapses_whitespace(self) -> None:
        values = pd.Series(["  Power   BI ", None, "SQL\nServer"])
        normalized = normalize_text(values)
        self.assertEqual(normalized.iloc[0], "Power BI")
        self.assertTrue(pd.isna(normalized.iloc[1]))
        self.assertEqual(normalized.iloc[2], "SQL Server")

    def test_column_profile_counts_nulls_and_blanks(self) -> None:
        frame = pd.DataFrame({"value": ["x", " ", None], "number": [1, 2, 3]})
        profile = {item["column"]: item for item in column_profile(frame)}
        self.assertEqual(profile["value"]["missing_or_blank"], 2)
        self.assertEqual(profile["number"]["missing_or_blank"], 0)

    def test_sha256_file_matches_hashlib(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.bin"
            source.write_bytes(b"SkillScope")
            self.assertEqual(sha256_file(source), hashlib.sha256(b"SkillScope").hexdigest())

    def test_expected_schema_has_unique_columns(self) -> None:
        self.assertEqual(len(EXPECTED_COLUMNS), 17)
        self.assertEqual(len(EXPECTED_COLUMNS), len(set(EXPECTED_COLUMNS)))


if __name__ == "__main__":
    unittest.main()
