from pydantic import BaseModel, Field, model_validator
from typing import Optional


class TableSuffix(BaseModel):
    """Suffixes for the underlying physical tables.

    `annual_and_metadata` is one parquet that holds every upgrade's annual
    results plus the building characteristics, with `upgrade=0` rows being
    the baseline. `timeseries` is the per-timestamp parquet, also covering
    every upgrade. Upgrade selection is a WHERE clause on the relevant
    table at query time — there is no separate baseline parquet.

    `annual_and_metadata_state_agg` (optional) names a *coarser-grained*
    alternative metadata table that the framework can route eligible
    queries to. It must contain the same logical content as
    `annual_and_metadata` but at state-or-coarser grain (e.g. tract
    columns absent, weights pre-summed across tract slices per state).
    Routing is automatic: when a query's `group_by` and `restrict`
    reference only columns present on the alt table,
    `_pick_metadata_table` picks `state_agg` and the query scans the
    smaller table. Schemas without this field always use
    `annual_and_metadata`.
    """

    annual_and_metadata: str
    timeseries: str
    annual_and_metadata_state_agg: Optional[str] = None


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
    # Column on the upgrade table that carries the human-readable upgrade name.
    # Defaults to the classic ResStock/ComStock convention. OEDI ComStock
    # overrides this to "in.upgrade_name". OEDI ResStock has no name column;
    # `get_upgrade_names` then degrades to NULL upgrade_name values.
    upgrade_name: str = "apply_upgrade.upgrade_name"
    fuel_totals: list[str]


class CompletionValues(BaseModel):
    success: str
    fail: str
    inapplicable: str


class Structure(BaseModel):
    # Vestigial: the codebase now assumes inapplicables_have_ts=True universally.
    # Kept here so existing TOMLs still validate; the value is ignored.
    inapplicables_have_ts: bool = True


class UniqueKeys(BaseModel):
    metadata: Optional[list[str]] = None
    timeseries: Optional[list[str]] = None
    # Unique-keys for the optional alt metadata table. Defaults to
    # [bldg_id] when omitted (one row per (bldg, state) is fine — the
    # alt table is *coarser* than primary, so its key is typically a
    # subset of primary's). No subset check vs metadata: the alt table
    # is allowed to have a narrower key.
    metadata_state_agg: Optional[list[str]] = None

    @model_validator(mode="after")
    def _timeseries_subset_of_metadata(self) -> "UniqueKeys":
        if self.metadata is not None and self.timeseries is not None:
            extra = set(self.timeseries) - set(self.metadata)
            if extra:
                raise ValueError(
                    "unique_keys.timeseries must be a subset of unique_keys.metadata; "
                    f"unexpected key(s): {sorted(extra)}"
                )
        return self


class DBSchema(BaseModel):
    table_suffix: TableSuffix
    column_prefix: ColumnPrefix
    column_names: ColumnNames
    completion_values: CompletionValues
    structure: Structure = Field(default_factory=Structure)
    unique_keys: UniqueKeys = Field(default_factory=UniqueKeys)
