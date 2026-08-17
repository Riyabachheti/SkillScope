# SkillScope India

SkillScope India is a data analytics and machine learning project based on Indian job postings. It cleans a public workbook, analyses skill demand with SQL, creates charts, and tests whether skills and job descriptions can distinguish four data-role families.

## Verified results

| Item | Result |
|---|---:|
| Source workbook | 97,929 rows / 17 columns |
| Final analytical data | 2,158 jobs / 16,778 job-skill pairs |
| Role families | 4 |
| Human-reviewed title sample | 157 titles |
| SQL analyses | 7, all matched independently with pandas |
| Charts | 5, each generated as PNG and SVG |
| Automated tests | 52 passing |
| Selected ML model | Skills + description Logistic Regression |
| Company-grouped OOF macro F1 | 0.781 |
| Accuracy | 0.830 |

## Data source and scope

The workbook matches the shape and fields of Shivam Shrivastava's [Indian Job Market Dataset 2025 (97k+ Data Points)](https://www.kaggle.com/datasets/shivamshrivastava21/indian-job-market-dataset-2025-2026), listed on Kaggle under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Kaggle does not provide a publisher checksum or a precise collection method, so the local file is described as *consistent with* the listing rather than as a proven copy. Its verified local SHA-256 is:

```text
bbcaaf9bb68465200b9090785426a542cd88af1b56326c29f8c1e7afd7949857
```

The results describe postings in this dataset, not the whole Indian job market. Important limitations include relative posting dates, repeated descriptions, generic skill tags, and many undisclosed salaries.

The raw workbook, processed tables, generated reports, charts, predictions, and trained model remain outside Git. SkillScope filters, labels, cleans, deduplicates, summarizes, and models the source; the publisher does not endorse these changes.

## Method

1. **Audit:** verify the workbook hash, worksheet, columns, missing values, duplicates, and general data quality.
2. **Role labels:** map titles to Data Analyst/BI, Data Scientist, Data Engineer, or ML/AI Engineer. A deterministic 157-title review reached 96.7–100% sampled precision across the four roles.
3. **Processing:** apply six reviewed title decisions, remove three repeated job IDs and 49 exact cross-ID copies, clean descriptions, normalize locations and skills, and retain template groups for ML splitting.
4. **Storage:** write one `jobs` table and one exploded `job_skills` table as Parquet.
5. **Analysis:** run seven DuckDB SQL queries and recalculate every result independently with pandas before creating charts.
6. **ML:** compare skills-only features with skills plus sanitized description TF-IDF using company-grouped cross-validation.

The final row count is fully accounted for:

```text
2,212 matched - 7 collisions + 5 reviewed additions - 3 repeated IDs - 49 exact copies = 2,158 jobs
```

## Reproduce the project

Place the unchanged workbook under `data/raw/`, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/audit_data.py
python scripts/generate_role_review.py
python scripts/validate_role_review.py
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/build_processed_data.py
python scripts/validate_processed_data.py
python scripts/run_analysis.py
python scripts/validate_analysis.py
python scripts/create_charts.py
python scripts/run_ml_experiment.py
python scripts/validate_ml.py
```

The expected result is 52 passing tests, 2,158 jobs, 16,778 job-skill pairs, seven SQL/pandas matches, five chart pairs, and validated ML predictions, metrics, folds, model selection, and model reload.

## Main findings

- **Role mix:** Data Engineer is the largest family with 1,055 postings (48.89%).
- **Overall skills:** Python appears in 53.15% of postings, SQL in 27.20%, and Machine Learning in 24.65%.
- **Role differences:** Analyst/BI postings often mention Data Analysis, SQL, Python, and Power BI. Data Engineer postings emphasize Python, SQL, Scala, and PySpark. Data Scientist and ML/AI postings both emphasize Python and Machine Learning.
- **Entry level:** Among 351 postings asking for 0–2 years of experience, Python appears in 51.00%, Machine Learning in 28.77%, SQL in 24.79%, and Data Analysis in 21.08%.
- **Locations:** Bengaluru, Hyderabad, and Pune are the largest first-listed physical locations; Data Engineer is the largest role family in each.
- **Skill pairs:** Python + SQL is the most common pair (17.93%), followed by Machine Learning + Python (15.20%) and PySpark + Python (12.88%). Association does not prove causation.
- **Company mix:** The largest normalized company label supplies 11.17% of retained postings, while the top ten supply 33.73%.

## Machine-learning evaluation

The role labels were created from job titles, so titles are excluded from the model to prevent target leakage.

Two feature sets are compared on the same five folds:

1. **Skills only:** one binary feature per normalized skill.
2. **Skills and description:** the same skill features plus word and two-word phrase TF-IDF. Direct role phrases such as `data engineer` are removed from a temporary modelling copy; the stored Parquet data is unchanged.

All postings from one company stay in the same fold, and repeated company-description templates are checked for separation. Feature vocabularies are fitted only on training rows.

| Model | OOF macro F1 | Accuracy |
|---|---:|---:|
| Majority baseline | 0.164 | 0.489 |
| Skills Logistic Regression | 0.751 | 0.805 |
| Skills Linear SVM | 0.710 | 0.776 |
| Skills + description Logistic Regression | **0.781** | **0.830** |
| Skills + description Linear SVM | 0.763 | 0.815 |

The largest remaining confusion is between Data Scientist and ML/AI Engineer. The validation script checks prediction coverage, company and template separation, recalculated metrics, model selection, and model reload. A second run produced the same metrics and prediction hashes.

These scores measure agreement with this project's reviewed labels on this dataset. They do not define job roles universally, prove cause and effect, or measure the whole Indian job market.
