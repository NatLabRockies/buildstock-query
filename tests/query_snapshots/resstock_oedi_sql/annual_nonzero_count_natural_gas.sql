SELECT baseline."in.geometry_building_type_recs" AS geometry_building_type_recs, sum(1) AS sample_count, sum(baseline.weight) AS units_count, sum(baseline."out.natural_gas.total.energy_consumption" * baseline.weight) AS "natural_gas.total.energy_consumption", sum(CASE WHEN (coalesce(baseline."out.natural_gas.total.energy_consumption", 0) != 0) THEN 1 ELSE 0 END * baseline.weight) AS "natural_gas.total.energy_consumption__nonzero_units_count" 
FROM (SELECT * 
FROM resstock_2024_amy2018_release_2_metadata 
WHERE resstock_2024_amy2018_release_2_metadata.upgrade = 0) AS baseline 
WHERE baseline.applicability = true AND baseline."in.state" = 'CO' GROUP BY 1 ORDER BY 1