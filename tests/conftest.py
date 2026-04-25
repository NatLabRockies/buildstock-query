from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from buildstock_query import BuildStockQuery

collect_ignore_glob = ["legacy/*"]

SNAPSHOTS_ROOT = Path(__file__).parent / "query_snapshots"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print real entry/variant totals — each pytest node processes a JSON file
    of N entries, so the test-function count understates actual coverage."""
    from tests.test_utility import SESSION_TOTALS

    if SESSION_TOTALS["entries"] == 0:
        return
    terminalreporter.write_sep("=", "snapshot entry totals")
    terminalreporter.write_line(
        f"  {SESSION_TOTALS['entries']} entries / {SESSION_TOTALS['variants']} variants"
    )
    terminalreporter.write_line(
        f"  passed={SESSION_TOTALS['passed']} updated={SESSION_TOTALS['updated']} "
        f"skipped={SESSION_TOTALS['skipped']} errored={SESSION_TOTALS['errored']} "
        f"failed={SESSION_TOTALS['failed']}"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--check-data",
        action="store_true",
        default=False,
        help="Run data comparison against parquet for entries whose SQL check failed.",
    )
    parser.addoption(
        "--update-snapshot",
        action="store_true",
        default=False,
        help=(
            "Auto-refresh the snapshot cache for changes the framework can prove are safe. "
            "Cosmetic SQL drift (sqlglot-equivalent) renames the parquet to the new hash with "
            "no data check. Real SQL drift auto-runs the query: if data matches → write the "
            "new pair and delete the old; if data is 'equivalent but different' (extra/missing "
            "columns with shared values agreeing) → write the new pair and delete the old; if "
            "data genuinely diverged → leave both alone (use --overwrite-snapshot to force)."
        ),
    )
    parser.addoption(
        "--overwrite-snapshot",
        action="store_true",
        default=False,
        help=(
            "Force-overwrite cache entries even when data genuinely diverged. Use only when "
            "you've deliberately changed query semantics. Includes everything --update-snapshot "
            "does."
        ),
    )


@pytest.fixture(scope="session")
def bsq_comstock_oedi() -> Iterator[BuildStockQuery]:
    print("\n[fixture] constructing BuildStockQuery(comstock_oedi)...", flush=True)
    bsq = BuildStockQuery(
        "rescore",
        "buildstock_sdr",
        "comstock_amy2018_r2_2025",
        buildstock_type="comstock",
        db_schema="comstock_oedi_state_and_county",
        skip_reports=True,
        cache_folder=str(SNAPSHOTS_ROOT / "comstock_oedi_cache"),
    )
    print("[fixture] comstock_oedi ready.", flush=True)
    yield bsq


@pytest.fixture(scope="session")
def bsq_resstock_oedi() -> Iterator[BuildStockQuery]:
    print("\n[fixture] constructing BuildStockQuery(resstock_oedi)...", flush=True)
    bsq = BuildStockQuery(
        "rescore",
        "buildstock_sdr",
        "resstock_2024_amy2018_release_2",
        buildstock_type="resstock",
        db_schema="resstock_oedi_vu",
        skip_reports=True,
        cache_folder=str(SNAPSHOTS_ROOT / "resstock_oedi_cache"),
    )
    print("[fixture] resstock_oedi ready.", flush=True)
    yield bsq
