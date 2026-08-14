WITH city_totals AS (
    SELECT primary_city, COUNT(*) AS city_postings
    FROM jobs
    WHERE primary_city <> 'Remote'
    GROUP BY primary_city
),
top_cities AS (
    SELECT primary_city, city_postings
    FROM city_totals
    ORDER BY city_postings DESC, primary_city
    LIMIT 10
),
city_roles AS (
    SELECT
        jobs.primary_city,
        jobs.role_category,
        COUNT(*) AS role_postings
    FROM jobs
    JOIN top_cities USING (primary_city)
    GROUP BY jobs.primary_city, jobs.role_category
)
SELECT
    city_roles.primary_city,
    top_cities.city_postings,
    city_roles.role_category,
    city_roles.role_postings,
    ROUND(100.0 * city_roles.role_postings / top_cities.city_postings, 2) AS within_city_pct
FROM city_roles
JOIN top_cities USING (primary_city)
ORDER BY top_cities.city_postings DESC, city_roles.primary_city,
         city_roles.role_postings DESC, city_roles.role_category;
