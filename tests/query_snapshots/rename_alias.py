"""Rewrite cached SQL + rehash + update manifests for the `_v__` → `ts__` rename.

Why this exists
---------------
Renaming an internal SELECT alias (`_v__<col>` → `ts__<col>`) is a pure
text-level change: the per-row scalar projected as `_v__foo` is now projected
as `ts__foo`, but the next layer's `SUM(...)` references it through the same
alias-lookup, and nothing outside the inner subquery sees the name. The
parquet result data is byte-identical; only the SQL text differs.

So instead of re-running 271 Athena queries for ~3 hours and ~$1.10, we:
  1. Find every cached `.sql` file containing `_v__`.
  2. Rewrite the SQL with `_v__` → `ts__` (whole-token replacement).
  3. Recompute the cache hash from the new SQL (sha256 over normalized text).
  4. Rename the .sql / .parquet / .json triple from <old_hash>.* → <new_hash>.*.
  5. Update every top-level snapshot manifest's `sql_hash` map so the entries
     still resolve to their (now-renamed) cache files.

Safety
------
- Refuses to overwrite an existing target hash (collision detection).
- Dry-run by default — pass `--apply` to actually rename.
- Verifies hash recomputation by reading back the renamed SQL.
- Never re-runs a query against Athena. Never modifies parquet content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

OLD = "_v__"
NEW = "ts__"

# Match the same way the framework does — see buildstock_query/sql_cache.py.
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_sql(sql: str) -> str:
    return _WHITESPACE_RE.sub(" ", sql).strip()


def hash_sql(sql: str) -> str:
    return hashlib.sha256(normalize_sql(sql).encode()).hexdigest()


def is_hash_name(s: str) -> bool:
    return len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def rewrite_sql(sql: str) -> str:
    """Substring substitution. The `_v__` prefix is unique to this convention
    (verified by grep across the snapshot corpus), so a bare str.replace is
    safe — no risk of clobbering an unrelated identifier."""
    return sql.replace(OLD, NEW)


def plan_cache_dir(cache_dir: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """Return (renames, errors) for one cache dir.

    renames: list of (old_hash, new_hash) for triples that need renaming.
    errors:  human-readable problems (collisions, missing files, etc.)."""
    renames: list[tuple[str, str]] = []
    errors: list[str] = []
    new_hashes_seen: dict[str, str] = {}  # new_hash -> old_hash that produced it

    for sql_path in sorted(cache_dir.glob("*.sql")):
        old_hash = sql_path.stem
        if not is_hash_name(old_hash):
            continue
        sql = sql_path.read_text()
        if OLD not in sql:
            continue

        # Sanity: the on-disk hash should match the on-disk SQL's hash. If it
        # doesn't, something else has already drifted and we shouldn't touch it.
        recomputed_old = hash_sql(sql)
        if recomputed_old != old_hash:
            errors.append(
                f"{sql_path.name}: filename hash {old_hash[:12]}… doesn't match "
                f"sha256(normalized SQL)={recomputed_old[:12]}…; skipping."
            )
            continue

        new_sql = rewrite_sql(sql)
        new_hash = hash_sql(new_sql)
        if new_hash == old_hash:
            errors.append(f"{sql_path.name}: rewrite produced identical hash; skipping.")
            continue

        # Collision check 1: two old hashes collapsing into the same new hash
        # (would mean the rename isn't bijective on this corpus).
        if new_hash in new_hashes_seen:
            errors.append(
                f"COLLISION: both {new_hashes_seen[new_hash][:12]}… and {old_hash[:12]}… "
                f"rewrite to {new_hash[:12]}…; skipping {old_hash[:12]}…."
            )
            continue
        new_hashes_seen[new_hash] = old_hash

        # Collision check 2: target files already exist on disk (would silently
        # overwrite a different cached query).
        target_sql = cache_dir / f"{new_hash}.sql"
        if target_sql.exists():
            errors.append(
                f"COLLISION: target {new_hash[:12]}….sql already exists in cache; "
                f"skipping rename of {old_hash[:12]}…."
            )
            continue

        renames.append((old_hash, new_hash))

    return renames, errors


def apply_renames(cache_dir: Path, renames: list[tuple[str, str]]) -> None:
    """Rewrite SQL text, then rename the triple. Order matters: write new
    SQL file FIRST, verify hash, then rename the parquet/json siblings,
    then delete the old SQL file. That way an interrupted run leaves the
    old cache intact alongside any partially-written new entries."""
    for old_hash, new_hash in renames:
        old_sql_path = cache_dir / f"{old_hash}.sql"
        new_sql_path = cache_dir / f"{new_hash}.sql"
        old_sql = old_sql_path.read_text()
        new_sql = rewrite_sql(old_sql)
        new_sql_path.write_text(new_sql)

        # Verify the new file actually hashes to the expected name.
        actual = hash_sql(new_sql_path.read_text())
        if actual != new_hash:
            new_sql_path.unlink()
            raise RuntimeError(
                f"Hash mismatch after writing {new_sql_path.name}: "
                f"got {actual[:12]}…, expected {new_hash[:12]}…"
            )

        # Move the parquet (always present alongside .sql in a real cache).
        old_pq = cache_dir / f"{old_hash}.parquet"
        new_pq = cache_dir / f"{new_hash}.parquet"
        if old_pq.exists():
            shutil.move(str(old_pq), str(new_pq))

        # Move the JSON metadata if it exists.
        old_json = cache_dir / f"{old_hash}.json"
        new_json = cache_dir / f"{new_hash}.json"
        if old_json.exists():
            shutil.move(str(old_json), str(new_json))

        # Now safe to delete the old SQL — its siblings are already moved.
        old_sql_path.unlink()


def update_manifest(manifest_path: Path, hash_map: dict[str, str], apply: bool) -> int:
    """Update `sql_hash` entries in a top-level snapshot manifest. Returns
    the number of replacements made (or that would be made in dry-run)."""
    try:
        data = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return 0
    if not isinstance(data, list):
        return 0

    replaced = 0

    def visit_sql_hash(entry: dict) -> None:
        nonlocal replaced
        sh = entry.get("sql_hash")
        if not isinstance(sh, dict):
            return
        for schema, value in list(sh.items()):
            if isinstance(value, str) and value in hash_map:
                sh[schema] = hash_map[value]
                replaced += 1

    for entry in data:
        if isinstance(entry, dict):
            visit_sql_hash(entry)
            # Some manifests nest entries under a "cases" or similar key — be
            # conservative and only walk one level, but check any dict children.
            for v in entry.values():
                if isinstance(v, list):
                    for sub in v:
                        if isinstance(sub, dict):
                            visit_sql_hash(sub)

    if apply and replaced:
        manifest_path.write_text(json.dumps(data, indent=2) + "\n")
    return replaced


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually rename files and update manifests. Default is dry-run.",
    )
    parser.add_argument(
        "--snapshots-root", type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to tests/query_snapshots/ (defaults to this script's directory).",
    )
    args = parser.parse_args()

    root = args.snapshots_root
    cache_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith("_cache"))
    if not cache_dirs:
        print(f"No *_cache/ dirs under {root}", file=sys.stderr)
        return 1

    # Phase 1: plan all renames per cache.
    all_hash_map: dict[str, str] = {}  # old_hash -> new_hash, across all caches
    per_cache: dict[str, list[tuple[str, str]]] = {}
    total_errors: list[str] = []

    for cache_dir in cache_dirs:
        renames, errors = plan_cache_dir(cache_dir)
        per_cache[cache_dir.name] = renames
        # If the same old_hash appears in two caches with the SAME SQL, both
        # rewrite to the same new_hash — that's not a collision, just a shared
        # entry. But if the same old_hash maps to two DIFFERENT new_hashes
        # (impossible if rewrite is deterministic, but defend anyway), flag it.
        for old, new in renames:
            if old in all_hash_map and all_hash_map[old] != new:
                total_errors.append(
                    f"{cache_dir.name}: {old[:12]}… maps to two different new "
                    f"hashes ({all_hash_map[old][:12]}… and {new[:12]}…)"
                )
            all_hash_map[old] = new
        total_errors.extend(f"{cache_dir.name}: {e}" for e in errors)

    # Print plan summary.
    print(f"{'Cache':<32}{'rename':>10}{'unaffected':>14}")
    print("-" * 56)
    grand_renames = 0
    for cache_dir in cache_dirs:
        renames = per_cache[cache_dir.name]
        total_sql = sum(1 for _ in cache_dir.glob("*.sql"))
        unaffected = total_sql - len(renames)
        print(f"{cache_dir.name:<32}{len(renames):>10}{unaffected:>14}")
        grand_renames += len(renames)
    print("-" * 56)
    print(f"{'TOTAL':<32}{grand_renames:>10}")
    print(f"\nDistinct old→new hash pairs across all caches: {len(all_hash_map)}")

    if total_errors:
        print("\nErrors / collisions:")
        for e in total_errors:
            print(f"  {e}")

    # Phase 2: scan top-level manifests for sql_hash entries needing updates.
    manifest_paths = sorted(root.glob("*.json"))
    manifest_changes: list[tuple[Path, int]] = []
    for mp in manifest_paths:
        n = update_manifest(mp, all_hash_map, apply=False)
        if n:
            manifest_changes.append((mp, n))

    print(f"\nManifest hash replacements (across {len(manifest_paths)} manifests):")
    total_manifest_repls = 0
    for mp, n in manifest_changes:
        print(f"  {mp.name}: {n} replacement(s)")
        total_manifest_repls += n
    print(f"  TOTAL: {total_manifest_repls}")

    if not args.apply:
        print("\n[DRY RUN] Re-run with --apply to perform the rename.")
        return 0

    if total_errors:
        print("\nRefusing to apply — fix errors above first.", file=sys.stderr)
        return 2

    # Phase 3: apply.
    print("\nApplying renames…")
    for cache_dir in cache_dirs:
        renames = per_cache[cache_dir.name]
        if not renames:
            continue
        apply_renames(cache_dir, renames)
        print(f"  {cache_dir.name}: renamed {len(renames)} triples")

    print("Updating manifests…")
    applied_manifest_repls = 0
    for mp in manifest_paths:
        n = update_manifest(mp, all_hash_map, apply=True)
        if n:
            print(f"  {mp.name}: {n} replacement(s)")
            applied_manifest_repls += n

    print(f"\nDone. {grand_renames} cache triples renamed, "
          f"{applied_manifest_repls} manifest entries updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
