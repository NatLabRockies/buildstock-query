SELECT baseline."in.state" AS state, sum(1) AS sample_count, sum(baseline.weight) AS units_count, sum(baseline."out.electricity.total.energy_consumption" * baseline.weight) AS "electricity.total.energy_consumption" 
FROM (SELECT * 
FROM resstock_2024_amy2018_release_2_metadata 
WHERE CAST(resstock_2024_amy2018_release_2_metadata.upgrade AS VARCHAR) = '0') AS baseline 
WHERE CAST(baseline.applicability AS VARCHAR) = 'true' AND baseline."in.state" IN ('CO', 'WY') GROUP BY 1 ORDER BY 1