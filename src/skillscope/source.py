"""Shared identity and schema contract for the manually downloaded workbook."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


EXPECTED_SOURCE_SHA256 = "bbcaaf9bb68465200b9090785426a542cd88af1b56326c29f8c1e7afd7949857"
EXPECTED_SHEET = "Sheet1"
EXPECTED_COLUMNS = (
    "title",
    "jobId",
    "currency",
    "jobUploaded",
    "companyName",
    "tagsAndSkills",
    "experience",
    "salary",
    "location",
    "companyId",
    "ReviewsCount",
    "AggregateRating",
    "jobDescription",
    "minimumSalary",
    "maximumSalary",
    "minimumExperience",
    "maximumExperience",
)


class SourceContractError(ValueError):
    """Raised when the local workbook is missing, ambiguous, or unexpected."""


@dataclass(frozen=True)
class SourceIdentity:
    path: Path
    sha256: str
    sheet: str
    columns: tuple[str, ...]


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_source_workbook(raw_dir: Path, source: Path | None = None) -> Path:
    """Resolve an explicit workbook or the only XLSX file in the raw directory."""
    if source is not None:
        resolved = source.expanduser().resolve()
        if not resolved.is_file():
            raise SourceContractError(f"Source workbook not found: {resolved}")
        if resolved.suffix.casefold() != ".xlsx":
            raise SourceContractError(f"Source workbook must be an .xlsx file: {resolved}")
        return resolved

    workbooks = sorted(raw_dir.glob("*.xlsx"))
    if len(workbooks) != 1:
        raise SourceContractError(
            f"Expected exactly one .xlsx file in {raw_dir}, found {len(workbooks)}"
        )
    return workbooks[0].resolve()


def validate_source_workbook(
    path: Path,
    expected_sha256: str = EXPECTED_SOURCE_SHA256,
) -> SourceIdentity:
    """Reject a workbook unless its checksum, sheet, and columns match the contract."""
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise SourceContractError(
            "Source checksum mismatch: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )

    with pd.ExcelFile(path, engine="openpyxl") as workbook:
        sheet_names = workbook.sheet_names
    if sheet_names != [EXPECTED_SHEET]:
        raise SourceContractError(
            f"Expected exactly ['{EXPECTED_SHEET}']; found {sheet_names}"
        )
    columns = tuple(
        pd.read_excel(path, sheet_name=EXPECTED_SHEET, nrows=0, engine="openpyxl").columns
    )
    if columns != EXPECTED_COLUMNS:
        raise SourceContractError(
            f"Source schema mismatch: expected {list(EXPECTED_COLUMNS)}, found {list(columns)}"
        )
    return SourceIdentity(path, actual_sha256, EXPECTED_SHEET, columns)
