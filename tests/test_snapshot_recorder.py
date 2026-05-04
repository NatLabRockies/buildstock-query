from __future__ import annotations

from types import SimpleNamespace

import sqlalchemy as sa

from tests.normalize_invariant_snapshot import _stable_key
from tests.snapshot_recorder import _jsonable_arg
from tests.test_utility import _rehydrate_args, _resolve_bsq_column_refs


def test_jsonable_arg_strips_metadata_alias_from_sa_columns():
    bs = sa.table(
        "metadata",
        sa.column("bldg_id"),
        sa.column("state"),
        sa.column("in.county_name"),
    ).alias("bs")

    actual = _jsonable_arg(
        {
            "group_by": [bs.c.bldg_id, bs.c["in.county_name"]],
            "restrict": [
                (bs.c.bldg_id, [1, 2]),
                ((bs.c.bldg_id, bs.c.state), [(1, "CO"), (2, "WY")]),
            ],
        }
    )

    assert actual == {
        "group_by": ["bldg_id", "in.county_name"],
        "restrict": [
            ["bldg_id", [1, 2]],
            [["bldg_id", "state"], [[1, "CO"], [2, "WY"]]],
        ],
    }


def test_rehydrate_args_strips_legacy_alias_prefixes_recursively():
    actual = _rehydrate_args(
        {
            "group_by": ["bs.bldg_id", "bs.in.county_name"],
            "restrict": [
                [["bs.bldg_id", "bs.state"], [[1, "CO"], [2, "WY"]]],
                ["bs.bldg_id", [1, 2]],
            ],
            "avoid": [["up.bldg_id", [3, 4]]],
        }
    )

    assert actual["group_by"] == ["bldg_id", "in.county_name"]
    assert actual["restrict"] == [
        (["bldg_id", "state"], [[1, "CO"], [2, "WY"]]),
        ("bldg_id", [1, 2]),
    ]
    assert actual["avoid"] == [("bldg_id", [3, 4])]


def test_normalize_stable_key_treats_alias_qualified_refs_as_same_shape():
    legacy = {
        "group_by": ["bs.bldg_id", "bs.in.county_name"],
        "restrict": [["bs.bldg_id", [1, 2]]],
    }
    canonical = {
        "group_by": ["bldg_id", "in.county_name"],
        "restrict": [["bldg_id", [1, 2]]],
    }

    assert _stable_key("resstock_oedi", "query", legacy) == _stable_key(
        "resstock_oedi", "query", canonical,
    )


def test_jsonable_arg_preserves_bsq_metadata_column_markers():
    bs = sa.table(
        "metadata",
        sa.column("bldg_id"),
        sa.column("state"),
        sa.column("county"),
    ).alias("bs")
    fake_bsq = SimpleNamespace(
        md_bldgid_column=bs.c.bldg_id,
        md_key_cols=(bs.c.bldg_id, bs.c.county, bs.c.state),
    )

    actual = _jsonable_arg(
        {
            "group_by": [fake_bsq.md_bldgid_column],
            "restrict": [
                (fake_bsq.md_bldgid_column, [1, 2]),
                (tuple(fake_bsq.md_key_cols), [(1, "001", "CO")]),
            ],
        },
        bsq=fake_bsq,
    )

    assert actual == {
        "group_by": [{"_bsq_col_ref": "md_bldgid_column"}],
        "restrict": [
            [{"_bsq_col_ref": "md_bldgid_column"}, [1, 2]],
            [{"_bsq_cols_ref": "md_key_cols"}, [[1, "001", "CO"]]],
        ],
    }


def test_rehydrate_round_trip_restores_live_bsq_metadata_columns():
    bs = sa.table(
        "metadata",
        sa.column("bldg_id"),
        sa.column("state"),
        sa.column("county"),
    ).alias("bs")
    fake_bsq = SimpleNamespace(
        md_bldgid_column=bs.c.bldg_id,
        md_key_cols=(bs.c.bldg_id, bs.c.county, bs.c.state),
        _get_column=lambda name, annual_only=True: {
            "bldg_id": bs.c.bldg_id,
            "county": bs.c.county,
            "state": bs.c.state,
        }[name],
    )
    recorded_args = {
        "group_by": [fake_bsq.md_bldgid_column],
        "restrict": [
            ("state", ["CO"]),
            (fake_bsq.md_bldgid_column, [1, 2]),
            (tuple(fake_bsq.md_key_cols), [(1, "001", "CO")]),
            (["bldg_id", "county", "state"], [[1, "001", "CO"]]),
        ],
    }

    replay_args = _resolve_bsq_column_refs(
        fake_bsq, _rehydrate_args(_jsonable_arg(recorded_args, bsq=fake_bsq)),
    )

    assert replay_args["group_by"][0] is fake_bsq.md_bldgid_column
    assert replay_args["restrict"][1][0] is fake_bsq.md_bldgid_column
    assert len(replay_args["restrict"][2][0]) == len(fake_bsq.md_key_cols)
    assert all(
        actual is expected
        for actual, expected in zip(replay_args["restrict"][2][0], fake_bsq.md_key_cols)
    )
    assert all(
        actual is expected
        for actual, expected in zip(replay_args["restrict"][3][0], fake_bsq.md_key_cols)
    )
