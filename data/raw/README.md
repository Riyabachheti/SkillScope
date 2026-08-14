# Raw data

The local source is an original, unedited workbook downloaded manually and kept outside Git.

## Recorded provenance

| Field | Value |
|---|---|
| Dataset | Indian Job Market Dataset 2025 (97k+ Data Points) |
| Publisher | Shivam Shrivastava |
| Source page | [Kaggle dataset listing](https://www.kaggle.com/datasets/shivamshrivastava21/indian-job-market-dataset-2025-2026) |
| Publisher-stated licence | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| Observed local filename | `indian-job-market-dataset-2025 2.xlsx` |
| Verified locally | 2026-08-14 |
| Size | 31,709,363 bytes |
| SHA-256 | `bbcaaf9bb68465200b9090785426a542cd88af1b56326c29f8c1e7afd7949857` |
| Workbook shape | 97,929 rows and 17 columns in `Sheet1` |

The Kaggle listing describes 97K+ Indian job listings and shows the same 17 fields found in the workbook. This agreement supports the attribution, but it is not cryptographic proof of the file's download origin because Kaggle does not provide a publisher checksum on the visible listing. The project therefore states that the local workbook is *consistent with* that listing rather than claiming independently verified chain-of-custody.

The licence is non-commercial and share-alike. Before distributing processed data or deploying an app that bundles it, preserve attribution and confirm that the intended use complies with the source licence.

Raw data files are intentionally excluded from Git. The publisher listing does not document a precise scrape date or collection query, so this project does not invent or repeat those details.

The pipeline does not depend on the observed filename. Put exactly one `.xlsx` workbook in this directory. Before reading its data, every source-facing command verifies the recorded SHA-256, the single `Sheet1` worksheet, and the exact ordered 17-column schema. A renamed copy is accepted; a changed or additional workbook is rejected.

Do not rename columns, edit rows, or overwrite the source file in place.
