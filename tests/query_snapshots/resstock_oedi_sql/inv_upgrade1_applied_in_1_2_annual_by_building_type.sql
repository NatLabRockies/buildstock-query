SELECT baseline."in.geometry_building_type_recs" AS geometry_building_type_recs, sum(1) AS sample_count, sum(baseline.weight) AS units_count, sum(upgrade."out.electricity.total.energy_consumption" * baseline.weight) AS "electricity.total.energy_consumption", sum(upgrade."out.natural_gas.total.energy_consumption" * baseline.weight) AS "natural_gas.total.energy_consumption" 
FROM (SELECT * 
FROM resstock_2024_amy2018_release_2_metadata 
WHERE resstock_2024_amy2018_release_2_metadata.upgrade = 0) AS baseline JOIN (SELECT * 
FROM resstock_2024_amy2018_release_2_metadata 
WHERE resstock_2024_amy2018_release_2_metadata.upgrade != 0) AS upgrade ON baseline.bldg_id = upgrade.bldg_id AND upgrade.upgrade = 1 AND upgrade.applicability = true 
WHERE baseline.applicability = true AND baseline."in.state" = 'CO' AND baseline.bldg_id IN (SELECT upgrade.bldg_id 
FROM (SELECT * 
FROM resstock_2024_amy2018_release_2_metadata 
WHERE resstock_2024_amy2018_release_2_metadata.upgrade != 0) AS upgrade 
WHERE upgrade.upgrade IN (1, 2) AND upgrade.applicability = true GROUP BY upgrade.bldg_id 
HAVING count(distinct(upgrade.upgrade)) = 2) GROUP BY 1 ORDER BY 1