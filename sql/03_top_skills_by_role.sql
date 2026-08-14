WITH role_totals AS (
    SELECT role_category, COUNT(*) AS denominator
    FROM jobs
    GROUP BY role_category
),
skill_counts AS (
    SELECT
        jobs.role_category,
        job_skills.skill_key,
        MIN(job_skills.skill) AS skill,
        COUNT(*) AS posting_count
    FROM jobs
    JOIN job_skills USING (job_id)
    GROUP BY jobs.role_category, job_skills.skill_key
),
ranked AS (
    SELECT
        skill_counts.*,
        role_totals.denominator,
        ROW_NUMBER() OVER (
            PARTITION BY skill_counts.role_category
            ORDER BY skill_counts.posting_count DESC, skill_counts.skill_key
        ) AS skill_rank
    FROM skill_counts
    JOIN role_totals USING (role_category)
)
SELECT
    role_category,
    skill_rank,
    skill,
    posting_count,
    denominator,
    ROUND(100.0 * posting_count / denominator, 2) AS penetration_pct
FROM ranked
WHERE skill_rank <= 10
ORDER BY role_category, skill_rank;
