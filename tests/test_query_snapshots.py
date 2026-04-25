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
    ],
)
def test_snapshot_flavor(request, schema, fixture_name, flavor):
    bsq = request.getfixturevalue(fixture_name)
    run_snapshot_file(SNAPSHOTS_ROOT / f"{flavor}.json", bsq, request.config, schema=schema)


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
        pytest.skip(f"no entries in {json_path}")

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
