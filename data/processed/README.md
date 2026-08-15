# Processed data

`scripts/build_processed_data.py` writes two local Parquet files here:

- `jobs.parquet`: one row per retained posting (2,158 jobs);
- `job_skills.parquet`: one row per unique job-skill pair (16,778 pairs).

The build includes the corrected duplicate rules and all six reviewed title decisions. These files stay outside Git and public deployment. Rebuild them locally from the separately downloaded workbook; do not force-add them.
