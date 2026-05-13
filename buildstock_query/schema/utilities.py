from __future__ import annotations

import datetime
from collections.abc import Callable, Sequence
from typing import Protocol

import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict, validate_call
from sqlalchemy.sql import sqltypes
from sqlalchemy.sql.elements import ColumnElement, KeyedColumnElement, Label
from sqlalchemy.sql.schema import Column, Table
from sqlalchemy.sql.selectable import Alias, FromClause, Select, SelectBase, Subquery

# from buildstock_query import BuildStockQuery  # can't import due to circular import


SqlColumn = Column | ColumnElement
SqlLabel = Label
ColumnExpression = SqlLabel | SqlColumn
# Alias is included so metadata role aliases flow through the same type guards
# as real tables.
TableHandle = Table | Subquery | Alias | FromClause
TableReference = TableHandle | str
SqlExpression = ColumnElement | KeyedColumnElement
SqlPredicate = ColumnElement
SqlFrom = FromClause
SelectQuery = Select
SqlFunction = Callable[..., SqlExpression]

# Backwards-compatible names retained for existing callers.
SACol = SqlColumn
SALabel = SqlLabel
DBColType = ColumnExpression
DBTableType = TableHandle
AnyTableType = TableReference
SQLExpression = SqlExpression
SQLClause = SqlPredicate
SQLFromClause = SqlFrom
SQLSelect = SelectQuery
SQLFunction = SqlFunction


class QueryCompiler(Protocol):
    def _compile(self, query: object) -> str:
        """Compile a SQLAlchemy expression to SQL text."""


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
    bsq: object | None = None
    name: str
    mapping_dict: dict
    key: ColumnExpression | str | Sequence[ColumnExpression | str]
    model_config = ConfigDict(arbitrary_types_allowed=True)


ColumnReference = ColumnExpression | str | MappedColumn
WeightSpec = ColumnReference | tuple[str, TableReference]
RestrictScalar = str | int | float | bool | datetime.date | datetime.datetime | None
RestrictValue = RestrictScalar | Sequence[RestrictScalar] | SelectBase | Subquery
RestrictColTuple = tuple[ColumnExpression, ...]
RestrictRowValue = tuple[RestrictScalar, ...]
RestrictCriteria = RestrictValue | Sequence[RestrictRowValue]
RestrictTuple = tuple[ColumnReference | RestrictColTuple, RestrictCriteria]

# Backwards-compatible public alias.
AnyColType = ColumnReference

validate_arguments = validate_call(config=ConfigDict(arbitrary_types_allowed=True))
