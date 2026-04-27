"""Local-only tests for pure-pandas report methods that consume full metadata parquets.

These methods (`report.get_applied_options`, `report.get_enduses_buildings_map_by_change`,
`report.get_options_report`) don't emit Athena SQL — they download full
metadata parquets from S3 via `download_metadata_and_annual_results` and
process them locally in pandas. Snapshot-style SQL hashing doesn't apply.

Run with `--include-local`. The first invocation downloads ~400 MB of
parquets to `tests/local_only/cache/` (gitignored).

Current state — KNOWN BROKEN ON OEDI ResStock:
   The OEDI ResStock metadata schema uses `out.<fuel>.<scope>.energy_consumption`
   (annual) and `in.<characteristic>` for inputs. Classic ResStock used
   `end_use_<fuel>_<scope>`, `fuel_use_<...>`, `apply_upgrade.option_<N>_name`,
   and `upgrade_costs.option_<N>_name`. These three report methods hardcode
   the classic-schema column patterns, so on OEDI they either:
     - find no matching columns → empty DataFrame → pandas length-mismatch crash
       (`get_applied_options`)
     - find no end-use columns → empty enduse list → trivially-empty result
       (`get_enduses_buildings_map_by_change`)

   Marked xfail(strict=True) so they go XPASS the moment the methods are
   ported to OEDI naming conventions (or schema-driven via column_names),
   forcing a deliberate review at that point.
"""
from __future__ import annotations

import pytest


# Three known-good resstock bldg_ids that have upgrade 1 applied in CO.
# Verified at test-design time via:
#   bsq.get_building_ids(applied_in=[1], restrict=[("state", ["CO"])])[:3]
KNOWN_BLDG_IDS = [13, 16, 33]


OEDI_INCOMPAT = pytest.mark.xfail(
    reason="Method hardcodes classic-ResStock column patterns "
           "(apply_upgrade.option_*_name, end_use_*, fuel_use_*) that don't "
           "exist on OEDI metadata. Will XPASS once ported to OEDI conventions.",
    strict=True,
)


@pytest.mark.local_only
@OEDI_INCOMPAT
def test_get_applied_options_returns_one_set_per_building(bsq_resstock_oedi_local):
    """get_applied_options(upgrade_id="1", bldg_ids=N) must return a list of N
    sets — one per building — each containing the option strings applied to
    that building under the upgrade. With include_base_opt=False (default),
    each entry is a set of strings; with include_base_opt=True, each entry
    is a dict.
    """
    bsq = bsq_resstock_oedi_local
    options = bsq.report.get_applied_options(upgrade_id="1", bldg_ids=KNOWN_BLDG_IDS)
    assert len(options) == len(KNOWN_BLDG_IDS), (
        f"expected {len(KNOWN_BLDG_IDS)} entries (one per bldg), got {len(options)}"
    )
    for i, opt_set in enumerate(options):
        assert isinstance(opt_set, set), f"entry {i} is {type(opt_set).__name__}, expected set"
    assert any(len(s) > 0 for s in options), (
        f"all {len(options)} buildings returned empty option sets — "
        "suspicious; upgrade 1 should name at least some options"
    )


@pytest.mark.local_only
@OEDI_INCOMPAT
def test_get_applied_options_with_base_opt_returns_dicts(bsq_resstock_oedi_local):
    """include_base_opt=True changes the per-bldg entry from set[str] to
    dict[option_str, baseline_char_value]. Pins the type-shape change."""
    bsq = bsq_resstock_oedi_local
    options = bsq.report.get_applied_options(
        upgrade_id="1", bldg_ids=KNOWN_BLDG_IDS, include_base_opt=True,
    )
    assert len(options) == len(KNOWN_BLDG_IDS)
    for i, entry in enumerate(options):
        assert isinstance(entry, dict), (
            f"entry {i} is {type(entry).__name__}, expected dict (include_base_opt=True)"
        )
        for k, v in entry.items():
            assert isinstance(k, str), f"option key not str: {type(k).__name__}"
            assert not isinstance(v, (list, dict, set)), (
                f"baseline value for option '{k}' is container {type(v).__name__}"
            )


@pytest.mark.local_only
@OEDI_INCOMPAT
def test_get_enduses_buildings_map_by_change_returns_subset_of_input(bsq_resstock_oedi_local):
    """get_enduses_buildings_map_by_change(upgrade_id="1", bldg_list=N) returns
    a dict[enduse_name, pd.Index[bldg_id]]. Each Index in the returned dict
    must be a subset of the input bldg_list — the method narrows down to
    buildings that exhibit the specified change for that enduse. At least
    one enduse should show change for at least one building (upgrade 1 is
    a real intervention, not a no-op).
    """
    bsq = bsq_resstock_oedi_local
    result = bsq.report.get_enduses_buildings_map_by_change(
        upgrade_id="1", change_type="changed", bldg_list=KNOWN_BLDG_IDS,
    )
    assert isinstance(result, dict), f"expected dict, got {type(result).__name__}"
    input_set = set(KNOWN_BLDG_IDS)
    for enduse, bldg_index in result.items():
        bldg_set = set(bldg_index.tolist())
        unexpected = bldg_set - input_set
        assert not unexpected, (
            f"enduse '{enduse}' returned bldg_ids {sorted(unexpected)} "
            f"that aren't in the input bldg_list {KNOWN_BLDG_IDS}"
        )
    assert any(len(idx) > 0 for idx in result.values()), (
        "no enduses showed any change for any of the 3 input buildings — "
        "suspicious for an actual upgrade"
    )
