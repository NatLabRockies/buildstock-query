from __future__ import annotations

from typing import Iterator

import pytest

from buildstock_query import BuildStockQuery

collect_ignore_glob = ["legacy/*"]


def pytest_addoption(parser):
    parser.addoption(
        "--check-data",
        action="store_true",
        default=False,
        help="Run data comparison against parquet for entries whose SQL check failed.",
    )
    parser.addoption(
        "--update-sql",
        action="store_true",
        default=False,
        help="Rewrite .sql files on mismatch (includes whitespace-only and sqlglot-only diffs).",
    )
    parser.addoption(
        "--update-sql-and-data",
        action="store_true",
        default=False,
        help=(
            "Rewrite .sql AND .parquet for entries where both SQL and data check failed. "
            "Implies --check-data and --update-sql."
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
    )
    print("[fixture] comstock_oedi ready.", flush=True)
    yield bsq
    try:
        print("\n[fixture] saving comstock_oedi query cache...", flush=True)
        bsq.save_cache()
        print("[fixture] comstock_oedi cache saved.", flush=True)
    except Exception as exc:
        print(f"[fixture] comstock_oedi save_cache failed (non-fatal): {exc}", flush=True)


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
    )
    print("[fixture] resstock_oedi ready.", flush=True)
    yield bsq
    try:
        print("\n[fixture] saving resstock_oedi query cache...", flush=True)
        bsq.save_cache()
        print("[fixture] resstock_oedi cache saved.", flush=True)
    except Exception as exc:
        print(f"[fixture] resstock_oedi save_cache failed (non-fatal): {exc}", flush=True)
