SELECT baseline.bldg_id 
FROM (SELECT * 
FROM resstock_2024_amy2018_release_2_metadata 
WHERE resstock_2024_amy2018_release_2_metadata.upgrade = 0) AS baseline 
WHERE baseline."in.state" = 'CO' AND baseline.bldg_id IN (SELECT upgrade.bldg_id 
FROM (SELECT * 
FROM resstock_2024_amy2018_release_2_metadata 
WHERE resstock_2024_amy2018_release_2_metadata.upgrade != 0) AS upgrade 
WHERE upgrade.upgrade IN (1) AND upgrade.applicability = true GROUP BY upgrade.bldg_id 
HAVING count(distinct(upgrade.upgrade)) = 1)