WITH total_jobs AS (
    SELECT COUNT(*) AS denominator FROM jobs
),
skill_counts AS (
    SELECT
        skill_key,
        MIN(skill) AS skill,
        COUNT(*) AS posting_count
    FROM job_skills
    GROUP BY skill_key
)
SELECT
    ROW_NUMBER() OVER (ORDER BY posting_count DESC, skill_key) AS skill_rank,
    skill,
    posting_count,
    total_jobs.denominator,
    ROUND(100.0 * posting_count / total_jobs.denominator, 2) AS penetration_pct
FROM skill_counts
CROSS JOIN total_jobs
ORDER BY skill_rank
LIMIT 20;
