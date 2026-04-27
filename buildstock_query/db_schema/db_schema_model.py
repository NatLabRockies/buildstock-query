from pydantic import BaseModel, Field, model_validator
from typing import Optional


class TableSuffix(BaseModel):
    """Suffixes for the underlying physical tables.

    Two shapes are supported:

    * 2-table (preferred, OEDI): set `annual_and_metadata` + `timeseries`. The
      annual_and_metadata table holds rows for every upgrade (including the
      baseline as `upgrade=0`); the upgrade selection happens as a WHERE
      clause on that single table at query time.
    * 3-table (legacy, classic schemas): set `baseline` + `upgrades` +
      `timeseries`. The baseline parquet and upgrades parquet are physically
      separate tables and joined explicitly.

    Exactly one of `{annual_and_metadata}` or `{baseline, upgrades}` must be
    provided.
    """

    timeseries: str
    annual_and_metadata: Optional[str] = None
    baseline: Optional[str] = None
    upgrades: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one_shape(self) -> "TableSuffix":
        has_unified = self.annual_and_metadata is not None
        has_split = self.baseline is not None or self.upgrades is not None
        if has_unified and has_split:
            raise ValueError(
                "table_suffix may set EITHER `annual_and_metadata` (2-table shape) "
                "OR both `baseline` and `upgrades` (3-table shape) — not both."
            )
        if not has_unified and not has_split:
            raise ValueError(
                "table_suffix must set either `annual_and_metadata` (2-table shape) "
                "or both `baseline` and `upgrades` (3-table shape)."
            )
        if has_split and (self.baseline is None or self.upgrades is None):
            raise ValueError(
                "table_suffix 3-table shape requires BOTH `baseline` and `upgrades`."
            )
        return self


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
