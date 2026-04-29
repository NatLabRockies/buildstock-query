# Tests guide

This directory uses three complementary test styles. Pick the right one when
adding a new test.

## Test layout

```
tests/
├── test_query_snapshots.py          SQL-hash + parquet pinning
├── query_snapshots/                  per-flavor JSON entries + content-addressed parquet cache
│   ├── *.json                          one file per "flavor" — annual, savings, timeseries, ...
│   └── *_oedi_cache/                   parquets keyed by sha256(SQL); shared across runs
├── test_invariants.py               cross-flow / cross-query / mutation invariants
├── test_schema_unique_keys.py       monkeypatch unit tests for composite-key plumbing
├── test_sql_cache.py                disk-cache contract tests
├── test_set_cover.py                avoid-restrict algorithm tests
├── test_query_core_cache_paths.py   cache path resolution
├── test_UpgradeAnalyzer.py          UpgradeAnalyzer YAML processing
├── conftest.py                       fixtures + CLI options
├── reference_files/                  fixtures for UpgradeAnalyzer + Viz
├── local_only/                       opt-in tests that download S3 parquets
│   ├── test_full_data_methods.py    pure-pandas report methods (xfail-pinned for OEDI)
│   └── cache/                        gitignored — populated by --include-local
├── legacy/                           Viz tests + helpers; collect_ignore_glob filters them out
└── utility.py                       snapshot harness internals (NOT a test file)
```

## When to add which kind of test

### Snapshot test (most queries)

Add an entry to a `query_snapshots/*.json` file when you're pinning a SQL shape
the library should keep emitting. The harness runs `get_query_only=True`,
hashes the SQL, looks up the parquet under that hash, and compares against the
stored result.

Use a snapshot when:
- The method has `get_query_only=True` (you can extract SQL without executing).
- The query bounds Athena cost: state-restricted (or partition-restricted),
  bounded `bldg_id` list, or annual-only.
- You want regression coverage for the SQL shape *and* the data the query
  produces.

### Invariant test (cross-flow / cross-query checks)

Add to `test_invariants.py` when you're asserting a relationship between
multiple queries — not the absolute output of one query. Examples:
- `annual = ts_year_collapse = sum(ts_monthly)` for the same group.
- `applied_in=[1,2]` returns the set intersection of `applied_in=[1]` and
  `applied_in=[2]`.
- A schema-mutation breaks the result by exactly the predicted factor.

Invariants catch bugs the snapshot harness can't, because they compare two
independent code paths against each other rather than comparing today's
result to a stored result. The snapshot harness can only tell you "this
matches what we recorded" — not "what we recorded is correct."

### Unit test (no Athena)

Add to `test_schema_unique_keys.py` (or a sibling new file) when you're
testing internal contracts that don't need a real database — schema
validation, query-model rejection, JOIN ON construction with synthetic
in-memory SA tables. Use `monkeypatch.setattr(BuildStockQuery, "_get_tables",
...)` to inject fake tables.

### Local-only test (rare)

Add to `local_only/` when the method:
- Doesn't emit Athena SQL (e.g. `report.get_applied_options` downloads full
  metadata parquets and processes in pandas).
- Requires the full data to test meaningfully.

Tag with `@pytest.mark.local_only`. The test is skipped by default; run with
`pytest --include-local`. Cache lives at `tests/local_only/cache/` (gitignored).

## Cost guardrails

**Athena costs add up fast on OEDI.** A single full-scan TS query is ~$16 at
$5/TB. Three of these in a row is a $50 accident.

Before adding a snapshot entry that uses `_method` (i.e. not `bsq.query()`),
inspect the generated SQL via `get_query_only=True` and verify:

1. Every reference to the timeseries table is constrained on **both**:
   - `state` (or another partition key), AND
   - `upgrade`

2. The `ts.upgrade = N` filter must be in the JOIN ON clause, not just in a
   baseline-side subquery WHERE — the TS table is still scanned across all
   upgrades before the join eliminates non-matching rows.

`bsq.query()` adds the `ts.upgrade = N` constraint automatically. Other entry
points may not. Audit the source if you're not sure.

### Known cost-trap methods

- `agg.get_building_average_kws_at` — has no `restrict` API at all; its
  TS-side join doesn't constrain `upgrade`. Even a 3-timestamp filter scans
  every upgrade × every building. Was responsible for an unintended $24 spend
  on 2026-04-25 (3.24 TB scan against resstock_oedi). The original snapshot
  entry was removed from `building_kws.json`; do NOT re-add until the method
  gains state/restrict arguments.

## Snapshot regeneration policy

Two flags control snapshot updates with different blast radius:

### `--update-snapshot`

Auto-refreshes entries the framework can prove are safe:
- **Cosmetic SQL drift** (sqlglot-equivalent): renames the parquet to the new
  hash with no data check.
- **Real SQL drift + data matches**: writes new pair, deletes old, patches JSON.
- **Real SQL drift + "equivalent but different" data** (extra/missing columns
  with shared values agreeing): writes new pair, deletes old, patches JSON.
- **Real SQL drift + data genuinely diverged**: leaves both alone. You need
  `--overwrite-snapshot` to force.

### `--overwrite-snapshot`

Force-overwrites cache entries even when data genuinely diverged. Use only
when you've **deliberately changed query semantics** and you've verified the
new numbers are correct. Includes everything `--update-snapshot` does.

### Cross-checking before `--overwrite-snapshot`

When you're about to overwrite snapshots whose data drifted (not just SQL
hash), first run any `test_invariants.py` cases that exercise the same query
shape. Decision rule:

- **Invariants green** → the new data is internally consistent across flows;
  overwrite is safe.
- **Invariants red** → don't overwrite. Investigate what regressed first.

This is how the 2026-04-26 `applied_only` TS-flow fix was validated:
`test_annual_equals_ts_year_equals_ts_monthly_sum[upgrade1_applied]` (which
had been xfail) flipped to green after the SQL fix, proving the new lower
totals matched the always-correct annual totals — only then was overwrite run.

## Mutation testing

When a test passes, it's worth asking: *would it fail if the thing it tests
were broken?* `test_comstock_composite_key_mutation_breaks_invariants`
demonstrates the pattern:

1. Construct a second `BuildStockQuery` with the schema deliberately mutated
   (e.g. drop `state` from `unique_keys`).
2. Run the same query that the canonical sibling test runs.
3. Assert the mutated result diverges from the canonical correct values by
   the expected factor (e.g. exact 46x inflation when the join cross-products
   across 46 metadata rows for `bldg_id=51037`).

If the mutation produces identical results, the test wasn't actually testing
what it claimed. This caught a real test-design flaw: an earlier composite-key
test used baseline-only annual queries (no JOIN) which the mutation couldn't
break — the test "passed" but was never actually exercising the keys.

When adding a new contract-pinning test, ask: *what's the smallest
schema/code change that would break this test?* If you can't think of one,
the test isn't pinning enough.

## Per-schema placeholders

Snapshot entries use `$ELECTRICITY_TOTAL`, `$BUILDING_TYPE_COL`, etc. The
resolver in `tests/utility.py` substitutes per-schema strings — `out.electricity.total.energy_consumption`
on resstock vs `...energy_consumption..kwh` on comstock. This lets one JSON
entry exercise both schemas without forking the args.

To restrict an entry to one schema (e.g. utility methods that need
`map_eiaid_column` which comstock TOMLs don't define):

```json
{
  "name": "...",
  "schemas": ["resstock_oedi"],
  ...
}
```

## Running tests

| Command | What it does |
|---|---|
| `pytest tests/` | Full suite excluding local-only (default) |
| `pytest tests/test_query_snapshots.py` | Just snapshot tests |
| `pytest tests/test_invariants.py` | Just invariants |
| `pytest tests/ --include-local` | Add local-only tests (downloads ~400 MB on first run) |
| `pytest tests/ --update-snapshot` | Refresh snapshots that drifted safely |
| `pytest tests/ --overwrite-snapshot` | Force-refresh including divergent data (read policy above first) |
| `pytest tests/ --check-data` | Run data check even when SQL matches |
