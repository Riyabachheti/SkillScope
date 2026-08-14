WITH entry_jobs AS (
    SELECT job_id
    FROM jobs
    WHERE experience_band = 'Entry level (0-2 years)'
),
entry_total AS (
    SELECT COUNT(*) AS denominator FROM entry_jobs
),
skill_counts AS (
    SELECT
        job_skills.skill_key,
        MIN(job_skills.skill) AS skill,
        COUNT(*) AS posting_count
    FROM entry_jobs
    JOIN job_skills USING (job_id)
    GROUP BY job_skills.skill_key
)
SELECT
    ROW_NUMBER() OVER (ORDER BY posting_count DESC, skill_key) AS skill_rank,
    skill,
    posting_count,
    entry_total.denominator,
    ROUND(100.0 * posting_count / entry_total.denominator, 2) AS penetration_pct
FROM skill_counts
CROSS JOIN entry_total
ORDER BY skill_rank
LIMIT 20;
