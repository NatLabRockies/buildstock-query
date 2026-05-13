from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeGuard, cast

import sqlalchemy as sa
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.selectable import SelectBase, Subquery

from buildstock_query.schema.utilities import (
    ColumnExpression,
    ColumnReference,
    RestrictColTuple,
    RestrictRowValue,
    RestrictTuple,
    RestrictValue,
    SelectQuery,
    SqlFrom,
    SqlLabel,
    SqlPredicate,
    TableReference,
    typed_literal,
)


class QueryFilterHost(Protocol):
    bs_table: SqlFrom

    def _get_column(
        self,
        column_name: ColumnReference,
        candidate_tables: Sequence[TableReference | None] | None = None,
        annual_only: bool = False,
    ) -> ColumnExpression:
        """Resolve a user-facing column reference to a SQL expression."""


@dataclass
class PropagatedPredicate:
    column_name: str
    criteria: object

    def clause_for(self, column: ColumnExpression) -> SqlPredicate | None:
        """Return this predicate rebound to a matching inner query column."""
        try:
            return QueryFilterMixin._literal_restrict_clause(column, self.criteria)
        except ValueError:
            return None


class QueryFilterMixin(QueryFilterHost):
    @staticmethod
    def _normalize_restrict_subquery(criteria: object, expected_width: int = 1) -> SelectQuery | None:
        """Return a select object when criteria is a width-compatible subquery."""
        if isinstance(criteria, SelectBase):
            if len(criteria.selected_columns) != expected_width:
                raise ValueError(
                    f"Subquery restrictions must select exactly {expected_width} column(s)."
                )
            return cast(SelectQuery, criteria)

        if isinstance(criteria, Subquery):
            if len(criteria.c) != expected_width:
                raise ValueError(
                    f"Subquery restrictions must select exactly {expected_width} column(s)."
                )
            return sa.select(*criteria.c)

        return None

    @staticmethod
    def _is_column_tuple(col_ref: object) -> TypeGuard[RestrictColTuple]:
        """Return true when a restriction targets multiple SQL columns."""
        if not isinstance(col_ref, tuple) or len(col_ref) == 0:
            return False
        return all(isinstance(c, (Column, SqlLabel)) for c in col_ref)

    def _multi_column_membership(
        self, col_ref: RestrictColTuple, criteria: RestrictValue | Sequence[RestrictRowValue]
    ) -> SqlPredicate:
        """Build an IN predicate for composite-key restriction criteria."""
        subquery = self._normalize_restrict_subquery(criteria, expected_width=len(col_ref))
        if subquery is not None:
            return sa.tuple_(*col_ref).in_(subquery)

        if self._is_literal_sequence(criteria):
            if not criteria:
                raise ValueError("Multi-column membership criteria cannot be empty.")
            for row in criteria:
                if not isinstance(row, tuple) or len(row) != len(col_ref):
                    raise ValueError(
                        f"Each row in multi-column criteria must be a tuple of length {len(col_ref)}."
                    )
            return sa.tuple_(*col_ref).in_(list(criteria))

        raise ValueError(
            "Multi-column restrict keys must be paired with a subquery or a sequence of row-tuples."
        )

    @staticmethod
    def _is_literal_sequence(criteria: object) -> TypeGuard[Sequence[object]]:
        """Return true for list-like literal criteria, excluding strings."""
        return isinstance(criteria, Sequence) and not isinstance(criteria, str)

    @staticmethod
    def _literal_restrict_clause(col: ColumnExpression, criteria: object) -> SqlPredicate:
        """Build a typed equality or IN predicate for literal criteria."""
        if QueryFilterMixin._is_literal_sequence(criteria):
            typed = [typed_literal(col, value) for value in criteria]
            if len(typed) > 1:
                return col.in_(typed)
            if len(typed) == 1:
                return col == typed[0]
            raise ValueError(f"Invalid criteria {criteria}")
        return col == typed_literal(col, criteria)

    @staticmethod
    def _literal_avoid_clause(col: ColumnExpression, criteria: object) -> SqlPredicate:
        """Build an inequality or NOT IN predicate for literal criteria."""
        if QueryFilterMixin._is_literal_sequence(criteria):
            if len(criteria) > 1:
                return col.not_in(criteria)
            if len(criteria) == 1:
                return col != criteria[0]
            raise ValueError(f"Invalid criteria {criteria}")
        return col != criteria

    def _get_restrict_clauses(
        self,
        restrict: Sequence[RestrictTuple],
        annual_only: bool = False,
        *,
        bs_table: SqlFrom | None = None,
    ) -> list[SqlPredicate]:
        """Convert restrict entries into SQLAlchemy WHERE predicates."""
        candidate_tables = (bs_table,) if bs_table is not None else None
        propagatable = self._collect_propagatable_predicates(restrict, bs_table=bs_table)

        clauses: list[SqlPredicate] = []
        for col_ref, criteria in restrict:
            if self._is_column_tuple(col_ref):
                clauses.append(self._multi_column_restrict_clause(col_ref, criteria, propagatable, bs_table))
                continue

            col = self._get_column(
                cast(ColumnReference, col_ref), candidate_tables=candidate_tables, annual_only=annual_only
            )
            subquery = self._normalize_restrict_subquery(criteria)
            if subquery is not None:
                subquery = self._inject_propagated(subquery, propagatable, {col.name}, bs_table=bs_table)
                clauses.append(col.in_(subquery))
            else:
                clauses.append(self._literal_restrict_clause(col, criteria))
        return clauses

    def _multi_column_restrict_clause(
        self,
        col_ref: RestrictColTuple,
        criteria: RestrictValue | Sequence[RestrictRowValue],
        propagatable: Sequence[PropagatedPredicate],
        bs_table: SqlFrom | None,
    ) -> SqlPredicate:
        """Inject pushdown predicates before building a composite-key restrict."""
        col_names = {c.name for c in col_ref if isinstance(c, Column)}
        if isinstance(criteria, SelectBase):
            criteria = self._inject_propagated(cast(SelectQuery, criteria), propagatable, col_names, bs_table=bs_table)
        elif isinstance(criteria, Subquery):
            criteria = self._inject_propagated(
                sa.select(*criteria.c), propagatable, col_names, bs_table=bs_table,
            )
        return self._multi_column_membership(col_ref, criteria)

    def _collect_propagatable_predicates(
        self,
        restrict: Sequence[RestrictTuple],
        *,
        bs_table: SqlFrom | None = None,
    ) -> list[PropagatedPredicate]:
        """Collect metadata predicates that can be pushed into subqueries."""
        target_bs = bs_table if bs_table is not None else self.bs_table
        candidate_tables = (target_bs,) if bs_table is not None else None
        out: list[PropagatedPredicate] = []
        if not restrict:
            return out

        for col_ref, criteria in restrict:
            if self._is_column_tuple(col_ref) or isinstance(criteria, (SelectBase, Subquery)):
                continue
            try:
                resolved_col = self._get_column(
                    cast(ColumnReference, col_ref), candidate_tables=candidate_tables, annual_only=True
                )
            except (ValueError, AttributeError):
                continue
            if not isinstance(resolved_col, Column) or resolved_col.table is not target_bs:
                continue

            out.append(PropagatedPredicate(resolved_col.name, criteria))
        return out

    def _inject_propagated(
        self,
        select: SelectQuery,
        propagatable: Sequence[PropagatedPredicate],
        scope_col_names: set[str],
        *,
        bs_table: SqlFrom | None = None,
    ) -> SelectQuery:
        """Add eligible outer predicates to a membership subquery."""
        if not propagatable:
            return select
        target_bs = bs_table if bs_table is not None else self.bs_table
        try:
            projected_names = {c.name for c in select.selected_columns}
        except AttributeError:
            return select

        target_names = projected_names & set(scope_col_names)
        for predicate in propagatable:
            if predicate.column_name not in target_names:
                continue
            inner_col = target_bs.c.get(predicate.column_name)
            if inner_col is None:
                continue
            clause = predicate.clause_for(inner_col)
            if clause is not None:
                select = select.where(clause)
        return select

    def _add_restrict(
        self,
        query: SelectQuery,
        restrict: Sequence[RestrictTuple],
        *,
        annual_only: bool = False,
        bs_table: SqlFrom | None = None,
    ) -> SelectQuery:
        """Append restrict predicates to a query when any are present."""
        if not restrict:
            return query
        return query.where(
            *self._get_restrict_clauses(restrict, annual_only=annual_only, bs_table=bs_table)
        )

    def _get_avoid_clauses(
        self, avoid: Sequence[RestrictTuple], *, annual_only: bool = False
    ) -> list[SqlPredicate]:
        """Convert avoid entries into SQLAlchemy WHERE predicates."""
        clauses: list[SqlPredicate] = []
        for col_ref, criteria in avoid:
            if self._is_column_tuple(col_ref):
                clauses.append(sa.not_(self._multi_column_membership(col_ref, criteria)))
                continue

            col = self._get_column(cast(ColumnReference, col_ref), annual_only=annual_only)
            subquery = self._normalize_restrict_subquery(criteria)
            if subquery is not None:
                clauses.append(col.not_in(subquery))
            else:
                clauses.append(self._literal_avoid_clause(col, criteria))
        return clauses

    def _add_avoid(
        self,
        query: SelectQuery,
        avoid: Sequence[RestrictTuple],
        *,
        annual_only: bool = False,
    ) -> SelectQuery:
        """Append avoid predicates to a query when any are present."""
        if not avoid:
            return query
        return query.where(*self._get_avoid_clauses(avoid, annual_only=annual_only))
