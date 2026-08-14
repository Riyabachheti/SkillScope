WITH role_counts AS (
    SELECT
        role_category,
        COUNT(*) AS posting_count
    FROM jobs
    GROUP BY role_category
)
SELECT
    role_category,
    posting_count,
    ROUND(100.0 * posting_count / SUM(posting_count) OVER (), 2) AS share_pct
FROM role_counts
ORDER BY posting_count DESC, role_category;
