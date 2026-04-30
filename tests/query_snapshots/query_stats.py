"""Summarize Athena query stats from cached snapshot metadata.

Walks every `*_cache/` directory under this folder, reads the per-query
`*.json` execution metadata, and prints a per-schema summary: query count,
bytes scanned, $ cost (at $5 / TB, decimal TB per AWS billing), and total
wall-clock time.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

USD_PER_TB = 5.0
BYTES_PER_TB = 10**12  # AWS bills on decimal TB, not 2**40
MS_PER_HOUR = 3_600_000


@dataclass
class SchemaStats:
    name: str
    queries: int = 0
    reused: int = 0  # result-cache hits — 0 bytes, 0 cost
    failed: int = 0  # status != SUCCEEDED, no Statistics block
    bytes_scanned: int = 0
    engine_ms: int = 0
    total_ms: int = 0
    queue_ms: int = 0
    by_substatement: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def billable_queries(self) -> int:
        return self.queries - self.reused - self.failed

    @property
    def cost_usd(self) -> float:
        return self.bytes_scanned / BYTES_PER_TB * USD_PER_TB

    @property
    def tb_scanned(self) -> float:
        return self.bytes_scanned / BYTES_PER_TB


def collect(cache_dir: Path) -> SchemaStats:
    stats = SchemaStats(name=cache_dir.name)
    for json_path in sorted(cache_dir.glob("*.json")):
        try:
            meta = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            continue

        stats.queries += 1
        substatement = meta.get("SubstatementType", "UNKNOWN")
        stats.by_substatement[substatement] += 1

        status = meta.get("Status", {}).get("State")
        if status != "SUCCEEDED":
            stats.failed += 1
            continue

        s = meta.get("Statistics", {})
        reused = s.get("ResultReuseInformation", {}).get("ReusedPreviousResult", False)
        if reused:
            stats.reused += 1
            # Reused results report 0 scanned bytes; still skip to be explicit.
            continue

        stats.bytes_scanned += s.get("DataScannedInBytes", 0)
        stats.engine_ms += s.get("EngineExecutionTimeInMillis", 0)
        stats.total_ms += s.get("TotalExecutionTimeInMillis", 0)
        stats.queue_ms += s.get("QueryQueueTimeInMillis", 0)
    return stats


def fmt_bytes(n: int) -> str:
    # Decimal units to match AWS billing convention.
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1000 or unit == "PB":
            return f"{n:,.2f} {unit}" if unit != "B" else f"{n:,} B"
        n /= 1000  # type: ignore[assignment]
    return f"{n:.2f} PB"


def fmt_ms(ms: int) -> str:
    if ms < 1000:
        return f"{ms} ms"
    if ms < 60_000:
        return f"{ms / 1000:.2f} s"
    if ms < MS_PER_HOUR:
        return f"{ms / 60_000:.2f} min"
    return f"{ms / MS_PER_HOUR:.2f} h"


def print_schema(stats: SchemaStats) -> None:
    print(f"\n=== {stats.name} ===")
    print(f"  Total query records:     {stats.queries:,}")
    print(f"    Successful & billable: {stats.billable_queries:,}")
    print(f"    Result-cache reused:   {stats.reused:,}")
    print(f"    Failed / non-success:  {stats.failed:,}")
    print(f"  Data scanned:            {fmt_bytes(stats.bytes_scanned)}  ({stats.tb_scanned:.4f} TB)")
    print(f"  Estimated cost @ $5/TB:  ${stats.cost_usd:,.2f}")
    print(f"  Engine execution time:   {fmt_ms(stats.engine_ms)}")
    print(f"  Total wall-clock time:   {fmt_ms(stats.total_ms)}")
    print(f"  Time spent queued:       {fmt_ms(stats.queue_ms)}")

    if stats.billable_queries:
        avg_bytes = stats.bytes_scanned / stats.billable_queries
        avg_ms = stats.total_ms / stats.billable_queries
        print(f"  Avg per billable query:  {fmt_bytes(int(avg_bytes))}, {fmt_ms(int(avg_ms))}")

    if stats.by_substatement:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(stats.by_substatement.items()))
        print(f"  Substatement types:      {breakdown}")


def main() -> None:
    root = Path(__file__).resolve().parent
    cache_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith("_cache"))

    if not cache_dirs:
        print(f"No *_cache directories found under {root}")
        return

    grand = SchemaStats(name="ALL SCHEMAS")
    for d in cache_dirs:
        s = collect(d)
        print_schema(s)
        grand.queries += s.queries
        grand.reused += s.reused
        grand.failed += s.failed
        grand.bytes_scanned += s.bytes_scanned
        grand.engine_ms += s.engine_ms
        grand.total_ms += s.total_ms
        grand.queue_ms += s.queue_ms
        for k, v in s.by_substatement.items():
            grand.by_substatement[k] += v

    print("\n" + "=" * 60)
    print_schema(grand)


if __name__ == "__main__":
    main()
