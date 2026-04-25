SELECT baseline."in.comstock_building_type" AS comstock_building_type, sum(1) AS sample_count, sum(baseline.weight) AS units_count, sum(baseline."out.natural_gas.total.energy_consumption..kwh" * baseline.weight) AS "natural_gas.total.energy_consumption..kwh", sum(CASE WHEN (coalesce(baseline."out.natural_gas.total.energy_consumption..kwh", 0) != 0) THEN 1 ELSE 0 END * baseline.weight) AS "natural_gas.total.energy_consumption..kwh__nonzero_units_count" 
FROM (SELECT * 
FROM comstock_amy2018_r2_2025_md_by_state_and_county_parquet 
WHERE comstock_amy2018_r2_2025_md_by_state_and_county_parquet.upgrade = 0) AS baseline 
WHERE baseline.applicability = true AND baseline.state = 'CO' GROUP BY 1 ORDER BY 1