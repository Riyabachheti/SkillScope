WITH total_jobs AS (
    SELECT COUNT(*) AS denominator FROM jobs
),
skill_counts AS (
    SELECT skill_key, MIN(skill) AS skill, COUNT(*) AS posting_count
    FROM job_skills
    GROUP BY skill_key
),
pair_counts AS (
    SELECT
        left_skill.skill_key AS skill_a_key,
        right_skill.skill_key AS skill_b_key,
        COUNT(*) AS pair_posting_count
    FROM job_skills AS left_skill
    JOIN job_skills AS right_skill
      ON left_skill.job_id = right_skill.job_id
     AND left_skill.skill_key < right_skill.skill_key
    GROUP BY left_skill.skill_key, right_skill.skill_key
),
ranked AS (
    SELECT
        ROW_NUMBER() OVER (
            ORDER BY pair_counts.pair_posting_count DESC,
                     pair_counts.skill_a_key,
                     pair_counts.skill_b_key
        ) AS pair_rank,
        skill_a.skill AS skill_a,
        skill_b.skill AS skill_b,
        pair_counts.pair_posting_count,
        skill_a.posting_count AS skill_a_postings,
        skill_b.posting_count AS skill_b_postings,
        total_jobs.denominator,
        ROUND(100.0 * pair_counts.pair_posting_count / total_jobs.denominator, 2) AS pair_penetration_pct,
        ROUND(
            pair_counts.pair_posting_count * total_jobs.denominator * 1.0
            / (skill_a.posting_count * skill_b.posting_count),
            3
        ) AS lift
    FROM pair_counts
    JOIN skill_counts AS skill_a ON pair_counts.skill_a_key = skill_a.skill_key
    JOIN skill_counts AS skill_b ON pair_counts.skill_b_key = skill_b.skill_key
    CROSS JOIN total_jobs
)
SELECT *
FROM ranked
ORDER BY pair_rank
LIMIT 20;
