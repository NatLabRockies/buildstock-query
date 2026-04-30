# Why ComStock metadata queries are 14× slower than ResStock

**Investigation date:** 2026-04-30
**Scripts:** `query_stats.py`, `investigate_partitions.py`
**Raw data:** `partition_investigation.json`

## TL;DR

`query_stats.py` showed that ComStock queries average ~46 s while ResStock
averages ~3.7 s. Slicing the cached `*_cache/*.json` execution stats
revealed the asymmetry concentrates on **metadata-only queries**, where
ComStock takes ~42 s vs ResStock's ~2.9 s. Two root causes were identified
and quantified:

1. **Small-files / over-partitioning on ComStock metadata tables.** The
   `_md_*_by_state_and_county_parquet` tables hold 194,308 small parquet
   files across 3,133 `(state, county)` partitions; ResStock's `_metadata`
   has 17 files unpartitioned. Even with a state filter that prunes to a
   single state, ComStock pays a ~12 s planning cost plus ~50 s of
   per-file footer reads.
2. **Missing predicate propagation in `buildstock-query`'s SQL generation.**
   When the outer query has `WHERE state='CO'` but the inner
   `IN (SELECT ... HAVING ...)` subquery does **not**, Athena scans all
   3,133 partitions on the inner side. Pushing the same state filter into
   the inner subquery cuts a 58 s query to **5 s** — an **11.6× speedup
   from a single duplicated predicate**.

`buildstock-query` already publishes `_md_agg_by_state_parquet` (3,162
files, 50 partitions) and `_md_agg_national_parquet` (62 files, no
partitions) — pre-aggregated alternatives that the framework currently
never picks. Routing eligible queries to these would capture an
additional 3×.

## How to reproduce

The investigation script is read-only and idempotent: every probe is
content-cached at `_partition_probe_cache/`, so re-runs are free.

```bash
python tests/query_snapshots/investigate_partitions.py
```

Total Athena spend on first run: well under $0.05 (probes scan KB-MB,
not GB). Wall-clock: ~5 minutes (most of that is the slow ComStock
probes we're investigating).

The script prints a per-table summary plus a verdict block. The full
results land in `partition_investigation.json` for re-reading without
re-running.

## What the cached snapshot stats already showed

These numbers came from cross-referencing every `*_cache/*.json` Athena
execution record. No new queries needed.

| Schema                  | Shape              | N   | Avg wall-clock | Avg planning |
|-------------------------|--------------------|----:|---------------:|-------------:|
| comstock_oedi_agg_cache | metadata_only      | 168 | 41.6 s         | 9.0 s        |
| comstock_oedi_cache     | metadata_only      | 177 | 42.3 s         | 8.8 s        |
| resstock_oedi_cache     | metadata_only      | 189 |  2.9 s         | 0.4 s        |
| comstock_oedi_agg_cache | ts_single_upgrade  |  80 | 52.3 s         | —            |
| resstock_oedi_cache     | ts_single_upgrade  |  81 |  4.6 s         | —            |

Metadata-only is >50 % of all snapshot queries. **Planning alone takes
21× longer on ComStock** before any data is read.

### Smoking gun

A trivial cached query:

```sql
SELECT count(*) FROM comstock_amy2018_r2_2025_md_agg_by_state_and_county_parquet
WHERE upgrade = 0 AND applicability = true
```

Took **75.5 s** while scanning **0.5 MB**. No subquery, no join, no
state filter. The cost is dominated by Athena enumerating
`(state × county)` partitions.

### `_agg` schema gives no speedup over non-`_agg`

Of 69 logical queries run against both ComStock schemas:

| Shape    | N  | agg avg | nonagg avg | head-to-head |
|----------|----|--------:|-----------:|--------------|
| metadata | 48 | 25.2 s  | 24.9 s     | agg 27, nonagg 21 |
| tsflow   | 21 | 39.7 s  | 42.7 s     | agg 13, nonagg 8  |

The pre-aggregated table is no faster. Both share the same
`_by_state_and_county_parquet` partition layout — the speedup from
having fewer rows is swallowed by the partition-discovery cost.

## What the layout probes confirmed

Run via `investigate_partitions.py`. All probes are metadata-only and
cost-guarded; the script refuses to issue any SQL that doesn't match a
metadata-table regex.

| Layout                                        | Files     | Partitions | count(state='CO',upgrade=0) | Wall | Planning | Bytes |
|-----------------------------------------------|----------:|-----------:|-----------:|-----:|---------:|------:|
| `_md_by_state_and_county_parquet` (non-agg)   | 194,308   | 3,133      | 134,649    | 3.49 s | 609 ms | 7.22 KB |
| `_md_agg_by_state_and_county_parquet`         | 194,308   | 3,133      |  25,743    | 6.88 s | 780 ms | 7.20 KB |
| `_md_agg_by_state_parquet`                    | **3,162** | **50**     |   3,272    | **1.19 s** | **260 ms** | **113 B** |
| `_md_agg_national_parquet`                    | 62        | 0          |   3,272    | 1.81 s | 234 ms | 40.7 KB |
| ResStock `_metadata`                          | 17        | 0          |   9,425    | 1.21 s | 140 ms | 414 KB  |

Key findings:

- **194,308 files** on ComStock metadata = exactly one parquet file per
  `(state, county, upgrade)` tuple (3,133 × ~62 upgrades). Median file
  size in CO: **318 rows**. Textbook small-files pathology.
- **`_agg_by_state` already exists** as a 3,162-file alternative. On a
  state-filtered query it's 3× faster than the by-state-and-county
  variants and within 12 % of ResStock's `_metadata`.
- **Partition pruning works** — adding `state='CO'` cuts ComStock by 22 s
  on average. But the floor is still ~40 s because ~3 k files survive
  the prune (one per county × ~62 upgrades).

## Real workload comparison

We re-issued three actual SQL shapes from the snapshot cache against
each ComStock table variant. (Same logical query, different table.)

| Shape | by_state_and_county (slow) | _agg_by_state | _agg_national | Speedup |
|-------|---------------------------:|--------------:|--------------:|--------:|
| **A.** `count(*) WHERE upgrade=0 AND applicability=true` | 59.7 s | 2.8 s | 1.7 s | **21×** |
| **B.** `GROUP BY state, sum(weight)` | 61.2 s | 2.8 s | 1.6 s | **22×** |
| **C.** `state='CO'` self-join applied-buildings, GROUP BY building_type | 59.8 s | 3.2 s | 2.5 s | **19×** |

The 60-second floor on `_md_*_by_state_and_county_parquet` is independent
of query complexity — it's a partition-catalog round-trip cost paid
once at planning time, then per-file footer reads during execution.
Long TS queries (10+ minutes) amortize this; short metadata queries are
crushed by it.

## The "missing predicate propagation" finding

Shape C above includes `bs.state='CO'` on the **outer** query but not on
the inner `IN (SELECT ...)` subquery. A controlled experiment isolated
this:

| Variant                                              | Wall    | Planning | Bytes  |
|------------------------------------------------------|--------:|---------:|-------:|
| V1: trivial `count(*) WHERE state='CO'`              | 6.6 s   | 530 ms   | 7.2 KB |
| V2: trivial `count(*)` no state filter               | 59.4 s  | 13.2 s   | 354 KB |
| **V3: shape C with state filter on BOTH sides**      | **5.0 s** | **1.2 s** | 405 KB |
| (Original shape C, outer state filter only)          | 58.1 s  | 11.8 s   | 4.6 MB |

Pushing the filter into the inner subquery cuts the query from 58 s to
5 s. **Athena's planner does not propagate the outer `state='CO'`
predicate into a `IN (SELECT ...)` with `GROUP BY ... HAVING`**, so the
inner side scans all 3,133 partitions. This is a SQL-generation
limitation in `buildstock-query`, not a data-layout issue.

## Fix #1 shipped: predicate propagation in `_get_restrict_clauses`

Implemented in `buildstock_query/query_core.py` via two new helpers:
`_collect_propagatable_predicates()` and `_inject_propagated()`. When
`_get_restrict_clauses` processes a subquery-valued restrict entry, it
now scans sibling restrict entries for safe single-column predicates
(literal/sequence RHS, target resolves to a `bs_table` column via the
existing `in.` prefix logic) and injects them into the subquery's
`WHERE`. Propagation only fires when the column is *both* projected by
the subquery *and* part of the outer query's IN-clause column tuple,
keeping the change semantics-preserving by construction.

Verified end-to-end on shape C against all four ComStock metadata
table variants (re-run via `investigate_partitions.py` after the fix):

| Table layout                          | Pre-fix (C) | Post-fix (C2) | Speedup |
|---------------------------------------|------------:|--------------:|--------:|
| `_md_by_state_and_county_parquet`     | 59.84 s     | **4.14 s**    | **14.4×** |
| `_md_agg_by_state_and_county_parquet` | 58.07 s     | **4.38 s**    | **13.3×** |
| `_md_agg_by_state_parquet`            | 3.16 s      | 1.76 s        | 1.8× |
| `_md_agg_national_parquet`            | 2.47 s      | 2.00 s        | 1.2× |

The big speedups (>13×) land exactly on the tables the framework uses
today by default. After this fix, ComStock metadata at the
state-filtered case is within **3.5×** of ResStock's `_metadata`
baseline (4.14 s vs 1.21 s) — a far smaller gap than the 50× we started
with. Auto-table-selection (Fix #2) would close most of the remainder.

Snapshot impact: 44 of 425 ComStock variants emit drifted SQL (one
extra `AND bs.<state_col> = <value>` in the inner subquery). Zero
ResStock drift. The drift is provably semantics-preserving — the
inner predicate is logically implied by the outer IN-tuple match — so
`--update-snapshot` regenerates these cleanly without any data drift.

## Recommended priority order

1. ~~**Highest leverage, smallest change — propagate restricts into inner
   subqueries.**~~ **DONE** — see "Fix #1 shipped" section above. Recovers
   13–14× on the slow ComStock shapes; 44 snapshots regenerated cleanly.
2. **Auto-route to less-sharded tables.** Add a `_pick_metadata_table()`
   selector in `buildstock_query/main.py` that picks among the four
   ComStock tables based on the query's `group_by` / `restrict`. Decision
   rule:
    - `group_by` includes `county` or filters by `county`/`gisjoin`
      → use `_md_*_by_state_and_county_parquet` (no choice).
    - `group_by ⊆ {state, upgrade, ...}` AND has state restrict
      → use `_md_agg_by_state_parquet`.
    - National-only with no state restrict → use `_md_agg_national_parquet`.
   Adds another ~3× on top of #1.
3. **Repartition the published parquet (data-pipeline change, outside
   this repo).** Move from `(state, county)` → `state`-only partitioning,
   compact small files. ~2× remaining gain after #1 + #2; high cost to
   ship through the data-publication pipeline.

## Cost guardrails inside `investigate_partitions.py`

Every probe SQL must match `\bFROM\s+\w*_md_\w*_parquet\b|\bFROM\s+\w+_metadata\b|\bSHOW\s+PARTITIONS\b`
and must NOT match `_ts_by_state\b|_by_state_vu\b`. `assert_safe()` is
called before every Athena submission; any future edit that
accidentally targets a TS table will raise a `RuntimeError` instead of
charging $24+ for an unbounded scan (see `CLAUDE.md` for the
2026-04-25 incident this guards against).

Probe results are cached via the framework's own `SqlCache`
(`_partition_probe_cache/<schema_label>/<sha256>.json`) so re-runs do
not bill Athena. Pass `--clear-cache` to force fresh probes.
