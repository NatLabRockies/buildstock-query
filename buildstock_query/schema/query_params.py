from collections.abc import Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from buildstock_query.schema.utilities import ColumnReference, RestrictTuple, TableReference, WeightSpec


class BaseQuery(BaseModel):
    enduses: Sequence[ColumnReference]
    group_by: Sequence[ColumnReference | tuple[str, str]] = Field(default_factory=list)
    upgrade_id: str = "0"
    sort: bool = True
    join_list: Sequence[tuple[TableReference, ColumnReference, ColumnReference]] = Field(default_factory=list)
    restrict: Sequence[RestrictTuple] = Field(default_factory=list)
    avoid: Sequence[RestrictTuple] = Field(default_factory=list)
    weights: Sequence[WeightSpec] = Field(default_factory=list)
    get_quartiles: bool = False
    get_nonzero_count: bool = False
    get_query_only: bool = False
    limit: int | None = None
    agg_func: str | None = "sum"
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", coerce_numbers_to_str=True)


class TSQuery(BaseQuery):
    timestamp_grouping_func: Literal["year", "month", "day", "hour"] | None = None


class UtilityTSQuery(TSQuery):
    query_group_size: int = 20
    eiaid_list: Sequence[str]


class Query(BaseQuery):
    annual_only: bool = True
    include_savings: bool = False
    include_baseline: bool = False
    include_upgrade: bool = True
    timestamp_grouping_func: Literal["year", "month", "day", "hour"] | None = None
    partition_by: Sequence[str] = Field(default_factory=list)
    applied_only: bool | None = Field(default=None)
    unload_to: str | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        effective_applied_only = self.upgrade_id != "0" if self.applied_only is None else self.applied_only
        if self.include_savings and self.upgrade_id == "0":
            raise ValueError("include_savings cannot be True when upgrade_id is '0'")
        if self.include_baseline and self.upgrade_id == "0":
            raise ValueError("include_baseline cannot be set when upgrade_id is '0'")
        if self.timestamp_grouping_func and self.annual_only:
            raise ValueError("annual_only must be False when timestamp_grouping_func is provided")
        if effective_applied_only and self.upgrade_id == "0":
            raise ValueError("applied_only cannot be set when upgrade_id is '0'")
        if self.get_nonzero_count and not self.annual_only:
            raise ValueError("get_nonzero_count cannot be True when annual_only is False")
        if self.get_quartiles and not self.annual_only:
            raise ValueError(
                "get_quartiles is not supported on timeseries queries (annual_only=False). "
                "Quartiles over per-timestamp rows don't compose meaningfully with a "
                "rollup, and quartiles over per-bucket sums are non-obvious; use "
                "min/max-style aggregates instead, or run an annual query for quartiles."
            )
        if self.applied_only is None:
            self.applied_only = effective_applied_only  # False for baseline, True otherwise
        return self
