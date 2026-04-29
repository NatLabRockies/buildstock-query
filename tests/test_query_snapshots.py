"""Snapshot-driven query tests.

Each test loads a shared JSON file at `tests/query_snapshots/<flavor>.json`,
runs every entry against the schema's session-scoped BuildStockQuery fixture
with `get_query_only=True`, compares the generated SQL to the stored
`<schema>_sql/<basename>.sql`, and (when `--check-data` is set) compares
returned data to `<schema>_data/<basename>.parquet`. Per-schema column-name
differences are expressed as `$ELECTRICITY_TOTAL` / `$BUILDING_TYPE_COL` /
etc. placeholders inside the JSON args; `tests/test_utility.py` resolves them
at load time based on the target schema.
"""
from __future__ import annotations

import pytest

from buildstock_query.schema.utilities import MappedColumn

from tests.test_utility import (
    SNAPSHOTS_ROOT,
    EntryOutcome,
    resolve_update_flags,
    evaluate_entries,
    format_failures,
    load_entries,
    run_snapshot_file,
)


SCHEMA_FIXTURES = [
    pytest.param("resstock_oedi", "bsq_resstock_oedi", id="resstock_oedi"),
    pytest.param("comstock_oedi", "bsq_comstock_oedi", id="comstock_oedi"),
    pytest.param("comstock_oedi_agg", "bsq_comstock_oedi_agg", id="comstock_oedi_agg"),
]


# --- generic per-flavor tests (args passed straight through to query()) -------

@pytest.mark.parametrize("schema, fixture_name", SCHEMA_FIXTURES)
@pytest.mark.parametrize(
    "flavor",
    [
        "annual",
        "timeseries",
        "savings",
        "applied_only",
        "restrict_avoid",
        "invariants_three_way",
        "building_ids",
        "helpers",
        "report",
        "utility",
        "building_kws",
    ],
)
def test_snapshot_flavor(request, schema, fixture_name, flavor):
    bsq = request.getfixturevalue(fixture_name)
    json_path = SNAPSHOTS_ROOT / f"{flavor}.json"
    if flavor == "utility":
        _rewrite_utility_entries(json_path, bsq, request.config, schema=schema)
        return
    run_snapshot_file(json_path, bsq, request.config, schema=schema)


# --- specialized: calculated column needs to be constructed live ------

@pytest.mark.parametrize("schema, fixture_name", SCHEMA_FIXTURES)
def test_calculated_column(request, schema, fixture_name):
    """JSON entries store `column_name`, `column_expr` (with placeholders),
    `table`, and the rest of the args. The test resolves placeholders inside
    the expression string, calls `bsq.get_calculated_column(...)` to build the
    SA expression, and passes it as the only enduse to `bsq.query()`."""
    bsq = request.getfixturevalue(fixture_name)
    from tests.test_utility import resolve_placeholder

    json_path = SNAPSHOTS_ROOT / "calculated_column.json"
    if not json_path.exists():
        pytest.skip(f"snapshot file not found: {json_path}")

    entries = load_entries(json_path, schema=schema)
    if not entries:
        pytest.skip(f"{json_path.name} has no entries for schema={schema} (all entries filtered out by their 'schemas' allowlist)")

    # The expression-side placeholder substitution happens here because the
    # generic loader resolves placeholders only for "$"-prefixed atomic
    # values, not embedded inside larger strings. We do the textual rewrite
    # manually so the user-facing JSON keeps the placeholder name.
    import re as _re
    placeholder_re = _re.compile(r"\$[A-Z_]+")

    for entry in entries:
        rewritten_variants = []
        for variant_args in entry.args:
            column_name = variant_args.pop("column_name")
            column_expr = variant_args.pop("column_expr")
            table = variant_args.pop("table", "baseline")
            # `annual=False` strips the `..kwh` suffix on comstock TS columns
            # — needed when `table='timeseries'` so the placeholder resolves
            # to the actual TS column name.
            placeholder_annual = table != "timeseries"
            def _sub(match, placeholder_annual=placeholder_annual):
                return resolve_placeholder(schema, match.group(0), annual=placeholder_annual)
            resolved_expr = placeholder_re.sub(_sub, column_expr)
            calculated_col = bsq.get_calculated_column(column_name, resolved_expr, table=table)
            rewritten_variants.append({**variant_args, "enduses": [calculated_col]})
        entry.args = rewritten_variants

    config = request.config
    check_data = config.getoption("--check-data")
    update_snapshot, overwrite_snapshot = resolve_update_flags(config)

    outcomes: list[EntryOutcome] = evaluate_entries(
        bsq, entries,
        check_data=check_data,
        update_snapshot=update_snapshot,
        overwrite_snapshot=overwrite_snapshot,
    )
    report = format_failures(outcomes, check_data=check_data)
    if report:
        pytest.fail(report, pytrace=False)


def _rewrite_utility_entries(json_path, bsq, config, *, schema):
    """utility.calculate_tou_bill needs a `rate_map` dict whose keys are
    (month, weekend, hour) tuples — JSON can't represent that directly. The
    JSON entry stores `rate_map_flat: <rate>` instead; here we expand it into
    the canonical flat-rate dict before invoking the harness."""
    if not json_path.exists():
        pytest.skip(f"snapshot file not found: {json_path}")

    entries = load_entries(json_path, schema=schema)
    if not entries:
        pytest.skip(f"{json_path.name} has no entries for schema={schema} (all entries filtered out by their 'schemas' allowlist — utility queries require map_eiaid_column, which only resstock TOMLs define)")

    for entry in entries:
        rewritten = []
        for variant in entry.args:
            if "rate_map_flat" in variant:
                flat_rate = variant.pop("rate_map_flat")
                variant["rate_map"] = {
                    (m, w, h): flat_rate
                    for m in range(1, 13) for w in (0, 1) for h in range(24)
                }
            rewritten.append(variant)
        entry.args = rewritten

    check_data = config.getoption("--check-data")
    update_snapshot, overwrite_snapshot = resolve_update_flags(config)

    outcomes: list[EntryOutcome] = evaluate_entries(
        bsq, entries,
        check_data=check_data,
        update_snapshot=update_snapshot,
        overwrite_snapshot=overwrite_snapshot,
    )
    report = format_failures(outcomes, check_data=check_data)
    if report:
        pytest.fail(report, pytrace=False)


# --- specialized: mapped_column needs to construct the MappedColumn live ------

@pytest.mark.parametrize("schema, fixture_name", SCHEMA_FIXTURES)
def test_mapped_column(request, schema, fixture_name):
    """Stored args carry `mapping_dict` and `key_column` (post-substitution);
    the test function constructs the MappedColumn and places it in either
    `enduses` or `group_by` based on `target`.
    """
    bsq = request.getfixturevalue(fixture_name)

    json_path = SNAPSHOTS_ROOT / "mapped_column.json"
    if not json_path.exists():
        pytest.skip(f"snapshot file not found: {json_path}")

    entries = load_entries(json_path, schema=schema)
    if not entries:
        pytest.skip(f"{json_path.name} has no entries for schema={schema} (all entries filtered out by their 'schemas' allowlist)")

    for entry in entries:
        rewritten_variants = []
        for variant_args in entry.args:
            target = variant_args.get("target", "group_by")  # 'group_by' or 'enduses'
            mapping_dict = variant_args["mapping_dict"]
            key_column = variant_args["key_column"]
            name = variant_args.get("mapped_name", "mapped_col")
            base_args = {
                k: v
                for k, v in variant_args.items()
                if k not in {"mapping_dict", "key_column", "mapped_name", "target"}
            }

            key_col = bsq._get_column(key_column)
            mapped = MappedColumn(bsq=bsq, name=name, mapping_dict=mapping_dict, key=key_col)
            if target == "group_by":
                base_args["group_by"] = [*base_args.get("group_by", []), mapped]
            elif target == "enduses":
                base_args["enduses"] = [*base_args.get("enduses", []), mapped]
            else:
                pytest.fail(f"{entry.name}: invalid target '{target}'")
            rewritten_variants.append(base_args)
        entry.args = rewritten_variants

    config = request.config
    check_data = config.getoption("--check-data")
    update_snapshot, overwrite_snapshot = resolve_update_flags(config)

    outcomes: list[EntryOutcome] = evaluate_entries(
        bsq,
        entries,
        check_data=check_data,
        update_snapshot=update_snapshot,
        overwrite_snapshot=overwrite_snapshot,
    )
    report = format_failures(outcomes, check_data=check_data)
    if report:
        pytest.fail(report, pytrace=False)
