from typing import Literal
from collections.abc import Sequence
import pandas as pd
import typing
from buildstock_query import main


class BuildStockAggregate:
    def __init__(self, buildstock_query: main.BuildStockQuery) -> None: ...

    @typing.overload
    def get_building_average_kws_at(
        self,
        at_hour: Sequence[float] | float,
        at_days: Sequence[float],
        enduses: Sequence[str],
        get_query_only: Literal[False] = False,
    ) -> pd.DataFrame: ...
    @typing.overload
    def get_building_average_kws_at(
        self,
        at_hour: Sequence[float] | float,
        at_days: Sequence[float],
        enduses: Sequence[str],
        get_query_only: Literal[True],
    ) -> str: ...

    def _query(self, *args, **kwargs) -> str | pd.DataFrame:
        ...
