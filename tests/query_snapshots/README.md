# Query snapshot tests

Snapshot-driven tests for `BuildStockQuery.query()`. Each JSON file stores a list of
curated query-argument sets. For every entry, the test harness:

1. Calls `query(**args, get_query_only=True)` against a session-scoped
   `BuildStockQuery` fixture connected to real Athena (`rescore.buildstock_sdr`).
2. Compares the generated SQL to the stored `.sql` file.
3. Optionally (`--check-data`) executes the query and compares the resulting DataFrame
   to the stored `.parquet`.

Failures from all entries in one file are collected and reported together.

## Layout

```
query_snapshots/
  comstock_oedi/      # comstock_amy2018_r2_2025 schema
    annual.json
    timeseries.json
    savings.json
    applied_only.json
    restrict_avoid.json
    sql/
      <sql_file>.sql
    data/
      <data_file>.parquet
  resstock_oedi/      # resstock_2024_amy2018_release_2 schema
    annual.json
    timeseries.json
    savings.json
    applied_only.json
    restrict_avoid.json
    mapped_column.json        # specialized — see below
    sql/
    data/
```

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
    "enduses": ["out.electricity.total.energy_consumption"],
    "group_by": ["state"],
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

# Rewrite .sql files for every SQL mismatch (whitespace-only, sqlglot-only, missing,
# structural). Use after a deliberate refactor where you've already verified the data
# is unchanged.
pytest -s -v tests/test_query_snapshots.py --update-sql

# Rewrite .sql AND .parquet for entries that fail BOTH checks. Implies
# --update-sql and --check-data. Use when a bugfix intentionally changes results.
pytest -s -v tests/test_query_snapshots.py --update-sql-and-data

# Run a single test function (fastest iteration):
pytest -s -v tests/test_query_snapshots.py::test_resstock_oedi_annual --update-sql-and-data
```

### Live progress

With `-s`, you'll see lines like:

```
[fixture] constructing BuildStockQuery(resstock_oedi)...
[fixture] resstock_oedi ready.

=== tests/query_snapshots/resstock_oedi/annual.json: 5 entries (check_data=True, update_sql=True, update_data=True) ===
[1/5] annual_totals_overall :: generating SQL...
[1/5] annual_totals_overall :: SQL generated in 0.12s
[1/5] annual_totals_overall :: status=missing_sql
[1/5] annual_totals_overall :: executing query for data check...
[1/5] annual_totals_overall :: data check finished in 14.33s (data_status=missing)
[1/5] annual_totals_overall :: wrote ['annual_totals_overall.sql', 'annual_totals_overall.parquet']
[2/5] annual_totals_by_building_type :: generating SQL...
...
```

### Flag matrix

| SQL check | `--check-data` | Data check | `--update-sql` | `--update-sql-and-data` | Action |
| --------- | -------------- | ---------- | -------------- | ----------------------- | ------ |
| pass      | —              | —          | —              | —                       | nothing |
| fail      | no             | —          | no             | no                      | report fail |
| fail      | yes            | pass       | no             | no                      | report "SQL changed, data OK" |
| fail      | yes            | fail       | no             | no                      | report both diffs |
| fail      | any            | any        | yes            | no                      | rewrite `.sql` |
| fail      | yes            | fail       | any            | yes                     | rewrite `.sql` AND `.parquet` |

## Typical workflows

### Bootstrapping a new entry

1. Add the entry to the appropriate JSON file (no `.sql` / `.parquet` yet).
2. Run `pytest tests/test_query_snapshots.py::<test_name> --update-sql-and-data`.
3. Commit the generated `.sql` and `.parquet`.

### After a SQL-generation refactor

1. Run `pytest tests/test_query_snapshots.py` — observe failures.
2. Run `pytest tests/test_query_snapshots.py --check-data` — confirm data still matches.
3. Run `pytest tests/test_query_snapshots.py --update-sql` to accept the SQL changes.
4. Review the resulting `.sql` diffs and commit.

### After a deliberate behavioral fix

1. Run `pytest tests/test_query_snapshots.py --check-data` — SQL mismatches with data
   mismatches means the fix changed results.
2. Confirm the new results are the correct ones (manually inspect).
3. Run `pytest tests/test_query_snapshots.py --update-sql-and-data`.
4. Review both the `.sql` and `.parquet` diffs carefully and commit.

## SQL comparison

1. **Exact match** — accept immediately.
2. **Whitespace-normalized match** — accept, treated as pass.
3. **sqlglot structural equivalence** — accept with `warnings.warn`.
4. **Otherwise** — report as a mismatch.

## Data comparison

Both expected and actual DataFrames are sorted by all columns (lexicographically)
before `pd.testing.assert_frame_equal` with global tolerances `rtol=1e-4`, `atol=1e-6`.

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
`--update-sql-and-data`. This enforces that every leg of every invariant is already
part of the snapshot suite.

Run with:

```bash
pytest -s -v tests/test_invariants.py
```

Included invariants (v1):

- `test_*_annual_equals_ts_year_equals_ts_monthly_sum` — per-group annual totals
  agree across annual-only, year-collapsed timeseries, and summed monthly timeseries.
- `test_*_savings_decomposition` — `baseline - upgrade ≈ savings` per group.

Tolerance: `rtol=1e-3, atol=1.0` (looser than the snapshot data check because
aggregate sums accumulate float error).
