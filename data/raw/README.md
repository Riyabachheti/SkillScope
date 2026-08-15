# Raw data

Keep the original workbook here without editing it. Raw data is not committed to Git.

## Source record

| Field | Value |
|---|---|
| Dataset | Indian Job Market Dataset 2025 (97k+ Data Points) |
| Publisher | Shivam Shrivastava |
| Source | [Kaggle dataset page](https://www.kaggle.com/datasets/shivamshrivastava21/indian-job-market-dataset-2025-2026) |
| Licence shown on Kaggle | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| Local filename used during verification | `indian-job-market-dataset-2025 2.xlsx` |
| Verified | 2026-08-14 |
| Size | 31,709,363 bytes |
| SHA-256 | `bbcaaf9bb68465200b9090785426a542cd88af1b56326c29f8c1e7afd7949857` |
| Shape | 97,929 rows and 17 columns in `Sheet1` |

The workbook has the same shape and fields as the Kaggle listing. Kaggle does not show a publisher checksum, so this match supports the attribution but does not prove where the local file was downloaded.

The current project keeps the workbook, processed records, generated outputs, and trained model local. A Streamlit app may be built and tested locally, but public deployment will wait until the licence position for that use is clear.

The publisher does not provide a precise collection date or search method, so the project does not invent those details.

The filename may change. Put exactly one `.xlsx` file in this directory. Before reading it, the pipeline checks the SHA-256, `Sheet1`, and all 17 columns. It accepts a renamed copy but rejects a changed file or a second workbook.

Do not rename columns, edit rows, or overwrite the workbook.
