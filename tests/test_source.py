from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from skillscope.source import (
    EXPECTED_COLUMNS,
    SourceContractError,
    resolve_source_workbook,
    sha256_file,
    validate_source_workbook,
)


class SourceContractTests(unittest.TestCase):
    def test_resolver_accepts_any_single_xlsx_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            workbook = raw_dir / "renamed-download.xlsx"
            workbook.write_bytes(b"fixture")
            self.assertEqual(resolve_source_workbook(raw_dir), workbook.resolve())

    def test_resolver_rejects_zero_or_multiple_workbooks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            with self.assertRaisesRegex(SourceContractError, "found 0"):
                resolve_source_workbook(raw_dir)
            (raw_dir / "first.xlsx").write_bytes(b"first")
            (raw_dir / "second.xlsx").write_bytes(b"second")
            with self.assertRaisesRegex(SourceContractError, "found 2"):
                resolve_source_workbook(raw_dir)

    def test_validator_rejects_checksum_mismatch_before_reading_excel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook = Path(temp_dir) / "wrong.xlsx"
            workbook.write_bytes(b"not-the-recorded-workbook")
            with self.assertRaisesRegex(SourceContractError, "checksum mismatch"):
                validate_source_workbook(workbook)

    def test_validator_accepts_expected_schema_and_rejects_schema_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            valid = Path(temp_dir) / "valid.xlsx"
            pd.DataFrame(columns=EXPECTED_COLUMNS).to_excel(valid, index=False)
            identity = validate_source_workbook(valid, expected_sha256=sha256_file(valid))
            self.assertEqual(identity.columns, EXPECTED_COLUMNS)

            invalid = Path(temp_dir) / "invalid.xlsx"
            changed_columns = [*EXPECTED_COLUMNS[:-1], "renamedExperience"]
            pd.DataFrame(columns=changed_columns).to_excel(invalid, index=False)
            with self.assertRaisesRegex(SourceContractError, "schema mismatch"):
                validate_source_workbook(invalid, expected_sha256=sha256_file(invalid))


if __name__ == "__main__":
    unittest.main()
