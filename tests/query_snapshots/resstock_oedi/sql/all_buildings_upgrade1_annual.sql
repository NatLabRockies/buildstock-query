SELECT baseline."in.geometry_building_type_recs" AS geometry_building_type_recs, sum(1) AS sample_count, sum(baseline.weight) AS units_count, sum(CASE WHEN (CAST(upgrade.applicability AS VARCHAR) = 'true') THEN upgrade."out.electricity.total.energy_consumption" ELSE baseline."out.electricity.total.energy_consumption" END * baseline.weight) AS "electricity.total.energy_consumption" 
FROM (SELECT * 
FROM resstock_2024_amy2018_release_2_metadata 
WHERE CAST(resstock_2024_amy2018_release_2_metadata.upgrade AS VARCHAR) = '0') AS baseline LEFT OUTER JOIN (SELECT * 
FROM resstock_2024_amy2018_release_2_metadata 
WHERE CAST(resstock_2024_amy2018_release_2_metadata.upgrade AS VARCHAR) != '0') AS upgrade ON baseline.bldg_id = upgrade.bldg_id AND CAST(upgrade.upgrade AS VARCHAR) = '1' AND CAST(upgrade.applicability AS VARCHAR) = 'true' 
WHERE CAST(baseline.applicability AS VARCHAR) = 'true' AND baseline."in.state" = 'CO' GROUP BY 1 ORDER BY 1