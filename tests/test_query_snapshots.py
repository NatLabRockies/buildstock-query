"""Snapshot-driven query tests.

Each test function loads a per-flavor JSON file, runs every entry against the session-
scoped BuildStockQuery fixture with `get_query_only=True`, compares the generated SQL to
the stored `.sql` file, and (when `--check-data` is set) also compares returned data to
the stored `.parquet` file. Failures are collected and reported together at the end.
"""
from __future__ import annotations

from buildstock_query.schema.utilities import MappedColumn

from tests.test_utility import (
    SNAPSHOTS_ROOT,
    EntryOutcome,
    evaluate_entries,
    format_failures,
    load_entries,
    run_snapshot_file,
)


# --- generic per-flavor tests (args passed straight through to query()) -------

def test_comstock_oedi_annual(bsq_comstock_oedi, request):
    run_snapshot_file(SNAPSHOTS_ROOT / "comstock_oedi" / "annual.json", bsq_comstock_oedi, request.config)


def test_comstock_oedi_timeseries(bsq_comstock_oedi, request):
    run_snapshot_file(SNAPSHOTS_ROOT / "comstock_oedi" / "timeseries.json", bsq_comstock_oedi, request.config)


def test_comstock_oedi_savings(bsq_comstock_oedi, request):
    run_snapshot_file(SNAPSHOTS_ROOT / "comstock_oedi" / "savings.json", bsq_comstock_oedi, request.config)


def test_comstock_oedi_applied_only(bsq_comstock_oedi, request):
    run_snapshot_file(SNAPSHOTS_ROOT / "comstock_oedi" / "applied_only.json", bsq_comstock_oedi, request.config)


def test_comstock_oedi_restrict_avoid(bsq_comstock_oedi, request):
    run_snapshot_file(SNAPSHOTS_ROOT / "comstock_oedi" / "restrict_avoid.json", bsq_comstock_oedi, request.config)


def test_resstock_oedi_annual(bsq_resstock_oedi, request):
    run_snapshot_file(SNAPSHOTS_ROOT / "resstock_oedi" / "annual.json", bsq_resstock_oedi, request.config)


def test_resstock_oedi_timeseries(bsq_resstock_oedi, request):
    run_snapshot_file(SNAPSHOTS_ROOT / "resstock_oedi" / "timeseries.json", bsq_resstock_oedi, request.config)


def test_resstock_oedi_savings(bsq_resstock_oedi, request):
    run_snapshot_file(SNAPSHOTS_ROOT / "resstock_oedi" / "savings.json", bsq_resstock_oedi, request.config)


def test_resstock_oedi_applied_only(bsq_resstock_oedi, request):
    run_snapshot_file(SNAPSHOTS_ROOT / "resstock_oedi" / "applied_only.json", bsq_resstock_oedi, request.config)


def test_resstock_oedi_restrict_avoid(bsq_resstock_oedi, request):
    run_snapshot_file(SNAPSHOTS_ROOT / "resstock_oedi" / "restrict_avoid.json", bsq_resstock_oedi, request.config)


# --- specialized tests (JSON stores partial args; test function completes the call) --

def test_resstock_oedi_mapped_column(bsq_resstock_oedi, request):
    """Stored args carry `mapping_dict` and `key_column`; the test function constructs the
    MappedColumn and places it in either `enduses` or `group_by` based on `target`.
    """
    import pytest

    json_path = SNAPSHOTS_ROOT / "resstock_oedi" / "mapped_column.json"
    if not json_path.exists():
        pytest.skip(f"snapshot file not found: {json_path}")

    entries = load_entries(json_path)
    if not entries:
        pytest.skip(f"no entries in {json_path}")

    # Build a derived entry list where each variant's stored args are rewritten to the
    # concrete `query()` kwargs after constructing the MappedColumn.
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

            key_col = bsq_resstock_oedi._get_column(key_column)
            mapped = MappedColumn(
                bsq=bsq_resstock_oedi, name=name, mapping_dict=mapping_dict, key=key_col
            )
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
    update_sql = config.getoption("--update-sql")
    update_data = config.getoption("--update-sql-and-data")
    if update_data:
        update_sql = True
        check_data = True

    outcomes: list[EntryOutcome] = evaluate_entries(
        bsq_resstock_oedi,
        entries,
        check_data=check_data,
        update_sql=update_sql,
        update_data=update_data,
    )
    report = format_failures(outcomes, check_data=check_data)
    if report:
        pytest.fail(report, pytrace=False)
