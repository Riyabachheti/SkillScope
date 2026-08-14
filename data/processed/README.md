# Processed data

`scripts/build_processed_data.py` writes two reproducible local artifacts here:

- `jobs.parquet`: one row per currently retained posting (2,158 after verified deduplication and title-review integration).
- `job_skills.parquet`: one row per unique normalized skill within a currently retained posting (16,778 pairs).

The cross-job deduplication fix and all six reviewed title decisions are verified. Generated artifacts are excluded from Git until remediation, licensing, and deployment-size decisions are complete. Rebuild them from the separately downloaded source workbook; do not force-add them.
