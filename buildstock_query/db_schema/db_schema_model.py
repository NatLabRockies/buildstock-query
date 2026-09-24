from pydantic import BaseModel
from typing import Optional


class TableSuffix(BaseModel):
    baseline: str
    timeseries: str
    upgrades: str


class ColumnPrefix(BaseModel):
    characteristics: str
    output: str


class ColumnNames(BaseModel):
    building_id: str
    sample_weight: str
    sqft: str
    timestamp: str
    completed_status: str
    unmet_hours_cooling_hr: str
    unmet_hours_heating_hr: str
    map_eiaid_column: Optional[str] = None  # Only for ResStock utility queries
    fuel_totals: list[str]


class CompletionValues(BaseModel):
    success: str
    fail: str
    inapplicable: str


class Structure(BaseModel):
    # whether the baseline timeseries is copied for unapplicable buildings in an upgrade
    inapplicables_have_ts: bool
    # Name of the geography column the baseline metadata is partitioned on, when that
    # metadata holds one row per (building, geography) rather than one row per building.
    # Runs whose dwelling units are allocated across geographies are published that way: a
    # building carries one weight per geography its units landed in, and its timeseries is
    # written once per geography as well. Joining such tables on building id alone gives a
    # cross product, so the join has to match on geography too.
    #
    # The value names the timeseries column (e.g. "state"); the baseline column is that
    # name behind the characteristics prefix (e.g. "in.state"). Leave unset for the usual
    # one-row-per-building publications, where building id alone is the right join.
    geography_partition: Optional[str] = None


class DBSchema(BaseModel):
    table_suffix: TableSuffix
    column_prefix: ColumnPrefix
    column_names: ColumnNames
    completion_values: CompletionValues
    structure: Structure
