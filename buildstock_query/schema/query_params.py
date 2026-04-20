from pydantic import ConfigDict, BaseModel, Field
from typing import Optional, Union
from collections.abc import Sequence
from typing import Literal
from buildstock_query.schema.utilities import AnyTableType, AnyColType
from pydantic import model_validator
from typing_extensions import Self


class BaseQuery(BaseModel):
    enduses: Sequence[AnyColType]
    group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list)
    upgrade_id: str = "0"
    sort: bool = True
    join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = Field(default_factory=list)
    restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list)
    avoid: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list)
    weights: Sequence[Union[str, tuple, AnyColType]] = Field(default_factory=list)
    get_quartiles: bool = False
    get_nonzero_count: bool = False
    get_query_only: bool = False
    limit: Optional[int] = None
    agg_func: Optional[str] = "sum"
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", coerce_numbers_to_str=True)


class TSQuery(BaseQuery):
    split_enduses: bool = False
    collapse_ts: bool = False
    timestamp_grouping_func: Optional[Literal["month", "day", "hour"]] = None


class SavingsQuery(TSQuery):
    annual_only: bool = True
    applied_only: bool = False
    unload_to: str = ""
    partition_by: Sequence[str] = Field(default_factory=list)


class UtilityTSQuery(TSQuery):
    query_group_size: int = 20
    eiaid_list: Sequence[str]


class Query(BaseQuery):
    annual_only: bool = True
    include_savings: bool = False
    include_baseline: bool = False
    include_upgrade: bool = True
    timestamp_grouping_func: Optional[Literal["year", "month", "day", "hour"]] = None
    partition_by: Sequence[str] = Field(default_factory=list)
    applied_only: Optional[bool] = Field(default=None)
    unload_to: Optional[str] = None

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.include_savings and self.upgrade_id == "0":
            raise ValueError("include_savings cannot be True when upgrade_id is '0'")
        if self.include_baseline and self.upgrade_id == "0":
            raise ValueError("include_baseline cannot be set when upgrade_id is '0'")
        if self.timestamp_grouping_func and self.annual_only:
            raise ValueError("annual_only must be False when timestamp_grouping_func is provided")
        if self.applied_only and self.upgrade_id == "0":
            raise ValueError("applied_only cannot be set when upgrade_id is '0'")
        if self.get_nonzero_count and not self.annual_only:
            raise ValueError("get_nonzero_count cannot be True when annual_only is False")
        if self.applied_only is None:
            self.applied_only = self.upgrade_id != "0"  # False for baseline, True otherwise
        return self
