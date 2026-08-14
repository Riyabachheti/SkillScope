WITH total_jobs AS (
    SELECT COUNT(*) AS denominator FROM jobs
),
company_counts AS (
    SELECT
        company_key,
        MIN(company_name) AS company_name,
        COUNT(*) AS posting_count,
        COUNT(DISTINCT role_category) AS represented_roles
    FROM jobs
    GROUP BY company_key
)
SELECT
    ROW_NUMBER() OVER (ORDER BY posting_count DESC, company_key) AS company_rank,
    company_name,
    posting_count,
    represented_roles,
    total_jobs.denominator,
    ROUND(100.0 * posting_count / total_jobs.denominator, 2) AS share_pct
FROM company_counts
CROSS JOIN total_jobs
ORDER BY company_rank
LIMIT 15;
