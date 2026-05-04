from __future__ import annotations

from tests.normalize_invariant_snapshot import _fill_blank_hashes


def test_fill_blank_hashes_backfills_only_missing_hashes(monkeypatch):
    existing_by_key = {
        "blank": {
            "name": "query__comstock_oedi__v1",
            "sql_hash": {"comstock_oedi": ""},
            "args": {"enduses": ["x"]},
        },
        "filled": {
            "name": "query__comstock_oedi__v2",
            "sql_hash": {"comstock_oedi": "already_there"},
            "args": {"enduses": ["y"]},
        },
    }

    def fake_signature(entry):
        if entry["name"].endswith("v1"):
            return "comstock_oedi", "query", {"enduses": ["x"]}
        return "comstock_oedi", "query", {"enduses": ["y"]}

    def fake_compute_cached_hash(*, schema, method, args_dict, bsq_by_schema):
        assert schema == "comstock_oedi"
        assert method == "query"
        if args_dict == {"enduses": ["x"]}:
            return "resolved_hash"
        return ""

    monkeypatch.setattr(
        "tests.normalize_invariant_snapshot._entry_signature",
        fake_signature,
    )
    monkeypatch.setattr(
        "tests.normalize_invariant_snapshot._compute_cached_hash",
        fake_compute_cached_hash,
    )

    filled = _fill_blank_hashes(existing_by_key)

    assert filled == 1
    assert existing_by_key["blank"]["sql_hash"]["comstock_oedi"] == "resolved_hash"
    assert existing_by_key["filled"]["sql_hash"]["comstock_oedi"] == "already_there"


def test_fill_blank_hashes_ignores_unresolvable_entries(monkeypatch):
    existing_by_key = {
        "blank": {
            "name": "query__comstock_oedi__v1",
            "sql_hash": {"comstock_oedi": ""},
            "args": {"enduses": ["x"]},
        },
    }

    monkeypatch.setattr(
        "tests.normalize_invariant_snapshot._entry_signature",
        lambda entry: ("comstock_oedi", "query", {"enduses": ["x"]}),
    )
    monkeypatch.setattr(
        "tests.normalize_invariant_snapshot._compute_cached_hash",
        lambda **kwargs: "",
    )

    filled = _fill_blank_hashes(existing_by_key)

    assert filled == 0
    assert existing_by_key["blank"]["sql_hash"]["comstock_oedi"] == ""
