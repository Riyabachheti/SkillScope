# SkillScope India

SkillScope India studies the skills requested in Indian data-job postings. It turns a public workbook into cleaned analytical tables, SQL findings, charts, and a role-classification experiment.

## Current status

- 2,158 cleaned job postings and 16,778 job-skill pairs;
- four reviewed role families;
- seven SQL analyses checked independently with pandas;
- five charts in PNG and SVG;
- 52 passing tests; and
- a leakage-aware ML experiment with repeatable results.

The selected model is Logistic Regression using skills and sanitized job descriptions. Its company-grouped out-of-fold macro F1 is **0.781**, compared with **0.751** for skills-only Logistic Regression and **0.164** for the majority baseline.

## Questions this project answers

- Which skills appear most often across data roles?
- How does skill demand differ between Data Analyst/BI, Data Scientist, Data Engineer, and ML/AI Engineer postings?
- Which skills commonly appear together?
- Which skills appear most often in entry-level postings?
- Can skills and job descriptions distinguish role families without using job titles as features?

## Scope and limits

The results describe postings in this dataset, not the whole Indian job market.

The source has relative posting dates, repeated descriptions, HTML, generic skill tags, and many undisclosed salaries. The pipeline measures and reports these limits instead of hiding them.

## Data source and licence

The workbook matches the shape and fields of Shivam Shrivastava's [Indian Job Market Dataset 2025 (97k+ Data Points)](https://www.kaggle.com/datasets/shivamshrivastava21/indian-job-market-dataset-2025-2026). Kaggle lists it under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Kaggle does not show a publisher checksum or a precise collection method. For that reason, this project says the local workbook is *consistent with* the listing; it does not claim a proven chain of custody or invent a collection date.

SkillScope filters, labels, cleans, deduplicates, summarizes, and models the source. These are changes made by this project, and the publisher does not endorse them.

### What is published

| Item | Current decision |
|---|---|
| Python, SQL, tests, reference configuration, and project method | May be published |
| README and summary results | May be published with source credit, licence link, changes, and limits |
| Raw workbook | Local only; download it from Kaggle |
| Processed Parquet and generated review samples | Local only |
| Generated CSV/JSON, charts, predictions, and trained model | Local only |
| Public Streamlit app | Not yet approved for deployment |

The original SkillScope code does not yet have a separate open-source licence. A public Streamlit app will be built and tested locally after the Phase 6 checkpoint. Public deployment will wait until the licence position for that specific use is clear.

## Verified source profile

| Check | Result |
|---|---:|
| Source rows | 97,929 |
| Source columns | 17 |
| Unique job IDs | 97,679 |
| Duplicate job-ID rows | 250 |
| Skills present | 99.417% |
| Salary not disclosed | 65.431% |
| Descriptions containing HTML | 71,744 |
| Postings matching at least one role rule | 2,212 |
| Hybrid collisions withheld | 7 |
| Unambiguous rule labels | 2,205 |
| Reviewed near misses added | 5 |
| Existing role labels corrected | 1 |
| Rule-or-review labels | 2,210 |
| Repeated job IDs removed | 3 |
| Exact cross-ID copies removed | 49 |
| Final jobs | 2,158 |
| Final job-skill pairs | 16,778 |
| Repeated template groups retained | 117 groups / 552 jobs |

The raw workbook stays outside Git. Its local SHA-256 is:

```text
bbcaaf9bb68465200b9090785426a542cd88af1b56326c29f8c1e7afd7949857
```

The filename may change, but `data/raw/` must contain exactly one `.xlsx` file. The pipeline checks its hash, `Sheet1`, and all 17 columns before reading it.

## Role labels

The source does not contain one clean role-category column. SkillScope creates four umbrella role families from job titles, then checks a deterministic sample of unique titles.

| Role | Correct / reviewed | Sampled precision |
|---|---:|---:|
| Data Analyst / BI | 30 / 30 | 100.0% |
| Data Scientist | 29 / 30 | 96.7% |
| Data Engineer | 30 / 30 | 100.0% |
| ML / AI Engineer | 30 / 30 | 100.0% |

Seven hybrid titles were withheld instead of being forced into one role. Manual review also added five missed titles and corrected one label. These six exact decisions are used by the transformation pipeline without making the general title rules broader.

```bash
python scripts/generate_role_review.py
python scripts/validate_role_review.py
```

The validator rejects missing decisions, an outdated sample, invalid labels, or any role below 90% sampled precision.

## Data processing

1. checks the workbook hash, sheet, and columns;
2. requires the current title review to pass;
3. applies the six reviewed title decisions;
4. keeps the most complete row for a repeated job ID;
5. removes a different job ID only when all 16 other source fields match;
6. keeps a template-group ID for later ML splitting;
7. removes HTML and redacts email addresses and Indian phone-like strings;
8. derives city, work arrangement, location count, and experience band;
9. standardizes skill aliases and creates one row per job-skill pair; and
10. checks schemas, keys, counts, text cleaning, and table relationships before writing Parquet.

```bash
python scripts/build_processed_data.py
python scripts/validate_processed_data.py
```

Validation confirms 2,158 unique jobs, 16,778 unique job-skill pairs, four roles, and zero orphan skill rows. The row count also balances:

```text
2,212 matched - 7 collisions + 5 reviewed additions - 3 repeated IDs - 49 exact copies = 2,158 jobs
```

Jobs that share a description template but differ in skills or experience are kept. Relative values such as `4 Days Ago` are also kept because the source does not provide a reliable reference date.

## Reproduce the project

From the repository root, run:

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

Expected result:

- 52 tests pass;
- the source and title-review checks pass;
- 2,158 jobs and 16,778 job-skill pairs are written;
- all seven SQL outputs match pandas;
- five PNG and five SVG charts are written; and
- ML folds, predictions, metrics, model selection, and model reload all validate.

## Analysis

Seven SQL files calculate role distribution, overall and role-specific skills, entry-level demand, city-role composition, skill pairs, and company concentration.

```bash
python scripts/run_analysis.py
python scripts/validate_analysis.py
```

The first command runs DuckDB directly over the Parquet files. The second calculates the same results with pandas and compares every row, value, and ordering rule.

### Main findings

- **Role mix:** Data Engineer is the largest family with 1,055 of 2,158 postings (48.89%). It is followed by ML/AI Engineer (19.32%), Data Scientist (16.36%), and Data Analyst/BI (15.43%).
- **Overall skills:** Python appears in 53.15% of postings, SQL in 27.20%, and Machine Learning in 24.65%. The denominator includes ten jobs without skill tags.
- **Role differences:** Analyst/BI postings often mention Data Analysis, SQL, Python, and Power BI. Data Engineer postings emphasize Python, SQL, Scala, and PySpark. Data Scientist and ML/AI postings both emphasize Python and Machine Learning.
- **Entry level:** Among 351 postings asking for 0–2 years of experience, Python appears in 51.00%, Machine Learning in 28.77%, SQL in 24.79%, and Data Analysis in 21.08%.
- **Locations:** Bengaluru, Hyderabad, and Pune are the largest first-listed physical locations. Data Engineer is the largest role family in each.
- **Skill pairs:** Python + SQL is the most common pair (17.93%), followed by Machine Learning + Python (15.20%) and PySpark + Python (12.88%). Association does not prove causation.
- **Company mix:** The largest normalized company label supplies 11.17% of retained postings; the top ten supply 33.73%. This concentration may influence the rankings.

Generic source tags such as `Data` remain in the reproducible outputs, but the summary above focuses on clearer tools and capabilities.

## Charts

```bash
python scripts/create_charts.py
```

The command checks that the analysis is current, then creates:

- role distribution;
- top skills by role;
- entry-level skills;
- city role composition; and
- skill co-occurrence.

Each chart is saved as PNG and SVG. `chart_manifest.json` records the input and output hashes. Generated charts stay outside Git and can be rebuilt locally.

## Machine-learning experiment

We test whether skills and job descriptions can distinguish the four reviewed role families. Because the target labels were created from job titles, titles are excluded from the model to prevent target leakage.

Two feature sets are compared using the same train/test splits:

1. **Skills only:** each normalized skill is a binary feature indicating whether it appears in the posting. The ten jobs without skill tags remain in the experiment as all-zero skill rows.
2. **Skills and description:** the binary skill features are combined with word and two-word phrase TF-IDF features from the cleaned description. Direct role phrases such as `data engineer` are removed from a temporary modelling copy before TF-IDF is applied. The stored Parquet data is not changed.

Evaluation uses five company-grouped folds. All postings from one company stay in the same fold, so the model is tested on companies it did not see during training. The code also checks that repeated company-description templates do not cross folds.

The skill and TF-IDF vocabularies are fitted only on the training portion of each fold. This prevents the held-out data from influencing feature construction.

```bash
python scripts/run_ml_experiment.py
python scripts/validate_ml.py
```

Each job receives one out-of-fold prediction while it is held out for testing. Macro F1 is the main metric because it gives equal importance to all four roles despite their different sample sizes.

The best result comes from class-balanced Logistic Regression using skills and sanitized descriptions:

| Model | OOF macro F1 | Accuracy |
|---|---:|---:|
| Majority baseline | 0.164 | 0.489 |
| Skills Logistic Regression | 0.751 | 0.805 |
| Skills Linear SVM | 0.710 | 0.776 |
| Skills + description Logistic Regression | **0.781** | **0.830** |
| Skills + description Linear SVM | 0.763 | 0.815 |

The largest remaining confusion is between Data Scientist and ML/AI Engineer. Their postings often share skills and language.

The validation script checks prediction coverage, fold separation, recalculated metrics, model selection, and model reload. A second run produced the same metrics and prediction hashes.

These scores measure agreement with this project's reviewed labels on this dataset. They do not define job roles universally, prove cause and effect, or measure the whole Indian job market.
