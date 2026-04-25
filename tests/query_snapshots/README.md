# Query snapshot tests

Snapshot-driven tests for `BuildStockQuery.query()`. Each shared JSON file at this
directory's top level stores a list of curated query-argument sets. For every entry,
the test harness:

1. Calls `query(**args, get_query_only=True)` against a session-scoped
   `BuildStockQuery` fixture connected to real Athena (`rescore.buildstock_sdr`).
2. Compares the generated SQL to the stored `.sql` file under `<schema>_sql/`.
3. Optionally (`--check-data`) executes the query and compares the resulting DataFrame
   to the stored `.parquet` under `<schema>_data/`.

Failures from all entries in one file are collected and reported together.

## Layout

```
query_snapshots/
  annual.json
  applied_only.json
  invariants_three_way.json
  mapped_column.json          # specialized — see below
  restrict_avoid.json
  savings.json
  timeseries.json
  resstock_oedi_sql/<basename>.sql
  resstock_oedi_data/<basename>.parquet
  comstock_oedi_sql/<basename>.sql
  comstock_oedi_data/<basename>.parquet
```

Each JSON entry runs once per schema. Per-schema column-name differences are written
as `$ELECTRICITY_TOTAL` / `$NATURAL_GAS_TOTAL` / `$BUILDING_TYPE_COL` / etc.
placeholders in the JSON args, resolved at load time by `tests/test_utility.py`'s
`SCHEMA_PLACEHOLDER_BUILDERS` map. Comstock annual columns get the `..kwh` suffix
that comstock's metadata table requires; comstock TS columns don't (the leg-aware
substitution keys off `annual_only`).

### Generic vs specialized

Generic JSON files (`annual.json`, etc.) store a full set of keyword args that are
passed straight through to `query()`. The test function just loops and compares.

Specialized JSON files (`mapped_column.json` today) store partial args; the
corresponding test function in `tests/test_query_snapshots.py` knows how to construct
the non-JSON-serializable pieces (e.g. a `MappedColumn` instance) from the stored
`mapping_dict` + `key_column`, then builds the final call.

### Entry shape

```json
{
  "name": "unique_entry_name",
  "description": "Human-readable purpose.",
  "sql_file": "unique_entry_name.sql",
  "data_file": "unique_entry_name.parquet",
  "args": {
    "enduses": ["$ELECTRICITY_TOTAL"],
    "group_by": ["$BUILDING_TYPE_COL"],
    "restrict": [["state", ["CO"]]]
  }
}
```

Tuples in `restrict`/`avoid` are stored as 2-element lists and re-hydrated to tuples
by the harness.

## Running

**Always pass `-s -v` when you want to see progress.** Without them pytest buffers
all output until the test finishes, which for long Athena queries looks like a frozen
run. `-s` disables stdout capture so the harness's per-entry `print()` lines show up
live; `-v` prints each test node name.

```bash
# Default: SQL-only comparison, fails on any mismatch.
pytest -s -v tests/test_query_snapshots.py

# Also run data check for entries whose SQL check failed; report only, no writes.
pytest -s -v tests/test_query_snapshots.py --check-data

# Routine refresh: write any change the framework can prove is safe — cosmetic SQL
# drift (whitespace_only, sqlglot_only) writes SQL with no Athena query; real SQL
# drift triggers a data check, then writes only if the new query produces matching,
# equivalent-shape, or missing-from-disk data. Real data divergence is left alone.
pytest -s -v tests/test_query_snapshots.py --update-snapshot

# Destructive: same as --update-snapshot plus also writes through real data
# mismatches. Use only when you've deliberately changed query semantics (bug fix,
# intentional output change) and the new value is the right value.
pytest -s -v tests/test_query_snapshots.py --overwrite-snapshot

# Run a single test function (fastest iteration):
pytest -s -v tests/test_query_snapshots.py::test_snapshot_flavor[annual-resstock_oedi] --update-snapshot
```

### Live progress

With `-s`, you'll see lines like:

```
[fixture] constructing BuildStockQuery(resstock_oedi)...
[fixture] resstock_oedi ready.

=== tests/query_snapshots/annual.json: 4 entries (check_data=True, update_snapshot=True, overwrite_snapshot=False) ===
[1/4] annual_totals_overall :: generating SQL...
[1/4] annual_totals_overall :: SQL generated in 0.12s
[1/4] annual_totals_overall :: status=missing_sql
[1/4] annual_totals_overall :: executing query for data check...
[1/4] annual_totals_overall :: data check finished in 14.33s (data_status=missing)
[1/4] annual_totals_overall :: wrote ['annual_totals_overall.sql', 'annual_totals_overall.parquet']
[2/4] annual_totals_by_building_type :: generating SQL...
...
```

### Decision matrix

| SQL status | Data status | `--update-snapshot` | `--overwrite-snapshot` |
| ---------- | ----------- | ------------------- | ---------------------- |
| match | (any) | nothing | nothing |
| whitespace_only | (skipped) | write SQL | write SQL |
| sqlglot_only | (skipped) | write SQL | write SQL |
| mismatch / missing_sql | match | write SQL | write SQL |
| mismatch / missing_sql | equivalent | write SQL + parquet | write SQL + parquet |
| mismatch / missing_sql | missing | write SQL + parquet | write SQL + parquet |
| mismatch / missing_sql | mismatch | leave both (failure) | write SQL + parquet |
| mismatch / missing_sql | error | leave both (failure) | leave both (failure) |

`equivalent` data status: column sets differ but every shared column matches within
tolerance — typically a refactor that adds or drops a metadata column without changing
the underlying query semantics.

## Typical workflows

### Bootstrapping a new entry

1. Add the entry to the appropriate JSON file (no `.sql` / `.parquet` yet).
2. Run `pytest tests/test_query_snapshots.py --update-snapshot`.
3. Commit the generated `.sql` and `.parquet`.

### After a SQL-generation refactor

1. Run `pytest tests/test_query_snapshots.py --update-snapshot`. The framework runs
   the data check internally and accepts the new SQL only if data still matches.
2. Review the resulting `.sql` diffs and commit. Any entry that didn't get written
   indicates a real data divergence — investigate before considering `--overwrite-snapshot`.

### After a deliberate behavioral fix

1. Run `pytest tests/test_query_snapshots.py --check-data`. Failures with `data_status`
   of `mismatch` indicate the fix changed results.
2. Confirm the new results are the correct ones (manually inspect the diff).
3. Run `pytest tests/test_query_snapshots.py --overwrite-snapshot`.
4. Review both the `.sql` and `.parquet` diffs carefully and commit.

## SQL comparison

1. **Exact match** — accept immediately.
2. **Whitespace-normalized match** — accept, treated as pass.
3. **sqlglot structural equivalence** — accept with `warnings.warn`.
4. **Otherwise** — report as a mismatch.

## Data comparison

Both expected and actual DataFrames are sorted by all columns (lexicographically)
before `pd.testing.assert_frame_equal` with global tolerances `rtol=1e-4`, `atol=1e-6`.

`is_data_equivalent_but_different` provides a looser secondary check: when the column
sets differ, the shared columns are compared on their own. If those agree, the entry
gets `data_status=equivalent` instead of `mismatch`, allowing `--update-snapshot` to
refresh both files.

## Invariant tests

`tests/test_invariants.py` asserts cross-query mathematical relations that must hold
regardless of implementation (e.g. annual total == sum of monthly timeseries totals).

Instead of issuing new Athena queries, invariants:

1. Construct `query()` arg sets for each leg.
2. Generate SQL via `get_query_only=True`.
3. Look up the snapshot entry whose stored `.sql` matches (normalized whitespace).
4. Load that entry's parquet and do the comparison locally.

If a required entry is missing from the JSON files or its parquet is absent, the
invariant fails with a message telling you to add the entry and re-run
`--update-snapshot`. This enforces that every leg of every invariant is already
part of the snapshot suite.

Run with:

```bash
pytest -s -v tests/test_invariants.py
```

Included invariants (v1):

- `test_annual_equals_ts_year_equals_ts_monthly_sum` — per-group annual totals
  agree across annual-only, year-collapsed timeseries, and summed monthly timeseries.
- `test_savings_decomposition` — `baseline - upgrade ≈ savings` per group.

Tolerance: `rtol=1e-3, atol=1.0` (looser than the snapshot data check because
aggregate sums accumulate float error).
