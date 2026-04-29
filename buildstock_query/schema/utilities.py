from __future__ import annotations
from typing import Union, Any
from collections.abc import Sequence
from pydantic import ConfigDict, BaseModel, validate_call
import sqlalchemy as sa
from sqlalchemy.sql import sqltypes
from sqlalchemy.sql.elements import Label, ColumnElement
from sqlalchemy.sql.selectable import SelectBase, Subquery, Alias
# from buildstock_query import BuildStockQuery  # can't import due to circular import


SACol = sa.Column | ColumnElement
SALabel = Label
DBColType = SALabel | SACol
# Alias is included so `bs_table` / `up_table` (SA aliases over the unified
# annual_and_metadata table) flow through the same type guards as real Tables.
DBTableType = sa.Table | Subquery | Alias
AnyTableType = Union[DBTableType, str]


def typed_literal(col, value):
    """Coerce a Python value to match the SQL type of `col`.

    Predicate pushdown on Athena/Trino requires comparing a column against a literal
    of matching type — `CAST(col AS VARCHAR) = '1'` defeats stripe pruning on parquet
    scans. Coercing the literal instead lets the column reference stay bare so the
    parquet reader can use min/max statistics to skip row groups.

    Pass-through for None and SQLAlchemy expressions (subqueries, other columns). For
    types we don't recognize, return the value unchanged.
    """
    if value is None:
        return value
    if hasattr(value, "__clause_element__") or isinstance(value, sa.sql.ClauseElement):
        return value
    col_type = getattr(col, "type", None)
    if col_type is None:
        return value
    if isinstance(col_type, sqltypes.Boolean):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "t", "1", "yes"}:
                return True
            if lowered in {"false", "f", "0", "no", ""}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return value
    if isinstance(col_type, (sqltypes.Integer, sqltypes.BigInteger, sqltypes.SmallInteger)):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if isinstance(col_type, sqltypes.String):
        return str(value) if not isinstance(value, str) else value
    return value


class MappedColumn(BaseModel):
    bsq: Any = None  # BuildStockQuery
    name: str
    mapping_dict: dict
    key: DBColType | str | Sequence[DBColType | str]
    model_config = ConfigDict(arbitrary_types_allowed=True)


AnyColType = DBColType | str | MappedColumn
RestrictValue = str | int | bool | Sequence[int | str | bool] | SelectBase | Subquery
RestrictColTuple = tuple[DBColType, ...]
RestrictRowValue = tuple[str | int | bool, ...]
RestrictTuple = (
    tuple[AnyColType, RestrictValue]
    | tuple[RestrictColTuple, SelectBase | Subquery | Sequence[RestrictRowValue]]
)

validate_arguments = validate_call(config=ConfigDict(arbitrary_types_allowed=True))
