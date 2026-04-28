# Query snapshot tests

Snapshot-driven tests for `BuildStockQuery.query()` (and a few other entry
points). Each shared JSON file at this directory's top level stores a list of
curated query-argument sets. Each entry carries an `sql_hash` field that
addresses the cached `<sql_hash>.sql` + `<sql_hash>.parquet` pair under
`<schema>_cache/`.

## How it works

The snapshot store IS the production cache, just frozen: each test fixture
constructs its `BuildStockQuery` with `cache_folder` pointing at
`<schema>_cache/`. Lookups happen automatically through the SqlCache layer in
`buildstock_query/sql_cache.py` — `bsq.query(...)` reads the parquet from
disk if it's there, falls through to Athena otherwise.

For each entry the harness:

1. Calls `query(**args, get_query_only=True)` to generate SQL.
2. Computes `hash_sql(actual_sql)`.
3. Compares the new hash to `entry.sql_hash`. Three outcomes: `match`,
   `sqlglot_only` (different hash, semantically equivalent), `mismatch`.
4. With `--check-data` or `--update-snapshot`, runs the query against Athena
   (or reads the cache if already present) and compares the DataFrame to the
   stored parquet.

## Layout

```
query_snapshots/
  annual.json
  applied_only.json
  building_ids.json
  invariants_three_way.json
  mapped_column.json          # specialized — see below
  restrict_avoid.json
  savings.json
  timeseries.json
  resstock_oedi_cache/<sha256>.sql
  resstock_oedi_cache/<sha256>.parquet
  comstock_oedi_cache/<sha256>.sql
  comstock_oedi_cache/<sha256>.parquet
```

Each JSON entry runs once per schema. Per-schema column-name differences are
written as `$ELECTRICITY_TOTAL` / `$NATURAL_GAS_TOTAL` / `$BUILDING_TYPE_COL`
/ etc. placeholders in the JSON args, resolved at load time by
`tests/test_utility.py`'s `resolve_placeholder()` dispatcher. Comstock annual
columns get the `..kwh` suffix; comstock TS columns don't (the leg-aware
resolution keys off `annual_only`).

### Generic vs specialized

Generic JSON files store full keyword args passed straight through to
`query()`. The test function loops and compares.

Specialized JSON files (`mapped_column.json` today) store partial args; the
corresponding test function in `tests/test_query_snapshots.py` knows how to
construct the non-JSON-serializable pieces (e.g. a `MappedColumn` instance)
from the stored `mapping_dict` + `key_column` fields.

### Entry shape

```json
{
  "name": "unique_entry_name",
  "sql_hash": "a3f2c1...",
  "description": "Human-readable purpose.",
  "args": {
    "enduses": ["$ELECTRICITY_TOTAL"],
    "group_by": ["$BUILDING_TYPE_COL"],
    "restrict": [["state", ["CO"]]]
  }
}
```

`sql_hash` is left as `""` for new entries — `--update-snapshot` fills it on
the next run. Tuples in `restrict`/`avoid` are stored as 2-element lists and
re-hydrated by the harness.

## Running

**Always pass `-s -v` when you want to see progress.** Without them pytest
buffers all output until the test finishes, which for long Athena queries
looks like a frozen run.

```bash
# Default: SQL-hash comparison only.
pytest -s -v tests/test_query_snapshots.py

# Also compare DataFrames against stored parquets.
pytest -s -v tests/test_query_snapshots.py --check-data

# Routine refresh: write any change the framework can prove is safe.
# Cosmetic SQL drift (sqlglot-equivalent) renames the parquet to the new hash
# without re-running the query. Real SQL drift triggers an Athena call;
# matching/equivalent/missing data → write new pair, delete old, patch JSON.
# Real data divergence is left alone.
pytest -s -v tests/test_query_snapshots.py --update-snapshot

# Destructive: same as --update-snapshot plus also writes through real data
# mismatches. Use only when query semantics changed deliberately.
pytest -s -v tests/test_query_snapshots.py --overwrite-snapshot
```

### Decision matrix

| SQL status   | Data status | `--update-snapshot`         | `--overwrite-snapshot`      |
|--------------|-------------|-----------------------------|-----------------------------|
| match        | (any)       | nothing                     | nothing                     |
| sqlglot_only | (skipped)   | rename parquet, patch JSON  | rename parquet, patch JSON  |
| mismatch     | match       | rename parquet, patch JSON  | rename parquet, patch JSON  |
| mismatch     | equivalent  | write new, delete old, patch| write new, delete old, patch|
| mismatch     | missing     | write new, delete old, patch| write new, delete old, patch|
| mismatch     | mismatch    | leave alone (failure)       | write new, delete old, patch|
| mismatch     | error       | leave alone (failure)       | leave alone (failure)       |

## Typical workflows

### Bootstrapping a new entry

1. Add the entry to the appropriate JSON file with `"sql_hash": ""`.
2. Run `pytest tests/test_query_snapshots.py --update-snapshot`.
3. Commit the generated `<hash>.sql` + `<hash>.parquet` and the patched JSON.

### After a SQL-generation refactor

1. Run `pytest tests/test_query_snapshots.py --update-snapshot`. Cosmetic
   drift gets a free rename; real drift triggers data verification.
2. Review `.sql` content diffs (the renames will appear as add+delete in git;
   `git diff --find-renames` or comparing the .sql file content tells you what
   really changed).

### After a deliberate behavioral fix

1. Run `pytest tests/test_query_snapshots.py --check-data`. `data_status` of
   `mismatch` indicates the fix changed results.
2. Confirm the new results are correct.
3. Run `pytest tests/test_query_snapshots.py --overwrite-snapshot`.
4. Review the parquet content carefully and commit.

## Invariant tests

`tests/test_invariants.py` asserts cross-query mathematical relations that
must hold regardless of implementation. Because the test fixtures point each
`BuildStockQuery` at the snapshot cache, invariants just call
`bsq.query(...)` directly — the cache returns the snapshot DataFrame.

If a required entry isn't in the cache (cold cache, or its hash doesn't match
the SQL the invariant generates), the invariant call falls through to Athena
— which works if you have credentials, fails otherwise. To avoid the network
round-trip, make sure the invariant's queries are present in one of the JSON
files and have populated hashes.

```bash
pytest -s -v tests/test_invariants.py
```

Tolerance: `rtol=1e-3, atol=1.0` (looser than the snapshot data check because
aggregate sums accumulate float error).

## Cleaning up stale cache entries

Each `<schema>_cache/` accumulates `<hash>.sql` / `<hash>.parquet` /
`<hash>.json` triples over time. When SQL generation evolves, old hashes
linger on disk even after their JSON entries point at new ones — the
update path writes the new pair but doesn't always know the old hash is
truly orphaned (the same hash might still be referenced by an invariant
test that bypasses the snapshot JSONs).

To resolve that ambiguity, `SqlCache` appends every read/write hash to
`.cache_usage_log` inside its `cache_folder`. The log is **append-only**
across BSQ constructions and pytest sessions, so multiple test runs
(snapshot suite + invariants) pool their hashes into a single record of
"what the test suite actually exercised". Truncate it explicitly when
you want to start a fresh tracking window — usually right before the
test runs you'll feed into cleanup.

`tests/cleanup_stale_caches.py` combines the JSON-referenced hashes with
the `.cache_usage_log` hashes to compute the set of cache files that are
still needed; everything else is stale.

```bash
# Step 1 — start a fresh tracking window.
python tests/cleanup_stale_caches.py --clear

# Step 2 — run the test sessions that should populate the log. Order
# doesn't matter; snapshot/invariants/etc. all append to the same log.
pytest -s -v tests/test_query_snapshots.py --check-data
pytest -s -v tests/test_invariants.py

# Step 3a — dry-run report.
python tests/cleanup_stale_caches.py
# Step 3b — actually remove stale files.
python tests/cleanup_stale_caches.py --delete
```

A wrapper script automates the whole flow:

```bash
# Clear → run snapshot+invariants → cleanup --delete.
python tests/refresh_caches.py

# Same, but report only (don't actually remove).
python tests/refresh_caches.py -n
```

**When to run:** after a behavioral fix or SQL refactor that changed
many hashes — those old `.parquet` files are megabyte-scale, so pruning
them keeps the repo from bloating. Always run the **invariants** suite
before cleanup, otherwise any cache entry that backs only an invariant
(not a snapshot JSON entry) will be classified stale and deleted; the
next invariants run will then have to re-execute those queries against
Athena.
