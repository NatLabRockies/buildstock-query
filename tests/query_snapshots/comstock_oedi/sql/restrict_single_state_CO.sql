SELECT baseline."in.comstock_building_type" AS comstock_building_type, sum(1) AS sample_count, sum(baseline.weight) AS units_count, sum(baseline."out.electricity.total.energy_consumption..kwh" * baseline.weight) AS "electricity.total.energy_consumption..kwh" 
FROM (SELECT * 
FROM comstock_amy2018_r2_2025_md_by_state_and_county_parquet 
WHERE CAST(comstock_amy2018_r2_2025_md_by_state_and_county_parquet.upgrade AS VARCHAR) = '0') AS baseline 
WHERE CAST(baseline.applicability AS VARCHAR) = 'true' AND baseline.state = 'CO' GROUP BY 1 ORDER BY 1