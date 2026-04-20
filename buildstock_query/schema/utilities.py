from __future__ import annotations
from typing import Union, Any
from collections.abc import Sequence
from pydantic import ConfigDict, BaseModel, validate_call
import sqlalchemy as sa
from sqlalchemy.sql.elements import Label, ColumnElement
from sqlalchemy.sql.selectable import Subquery
# from buildstock_query import BuildStockQuery  # can't import due to circular import


SACol = sa.Column | ColumnElement
SALabel = Label
DBColType = SALabel | SACol
DBTableType = sa.Table | Subquery
AnyTableType = Union[DBTableType, str]


class MappedColumn(BaseModel):
    bsq: Any = None  # BuildStockQuery
    name: str
    mapping_dict: dict
    key: DBColType | str | Sequence[DBColType | str]
    model_config = ConfigDict(arbitrary_types_allowed=True)


AnyColType = DBColType | str | MappedColumn

validate_arguments = validate_call(config=ConfigDict(arbitrary_types_allowed=True))
