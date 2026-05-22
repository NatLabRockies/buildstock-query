"""
Cleanup script for AWS Athena databases.

Connects to an Athena database/workgroup via boto3 and identifies tables/views
whose underlying S3 data no longer exists or that return zero rows, then
optionally drops them.

Usage:
    aws_athena_cleanup -d my_db -w primary -r us-west-2
    aws_athena_cleanup -d my_db -w primary --drop
"""

import argparse
from typing import Optional

from .aws_utils import (
    get_clients,
    wait_for_query,
    start_query,
    run_query,
    list_tables,
    list_views,
    get_table_s3_location,
    s3_path_has_data,
)


def table_has_rows(
    athena_client, database: str, workgroup: str, table_name: str, s3_output: Optional[str] = None
) -> bool:
    """Check if a table returns at least one row."""
    try:
        rows = run_query(
            athena_client,
            f'SELECT 1 FROM "{table_name}" LIMIT 1',
            database,
            workgroup,
            s3_output,
        )
        # rows includes header row, so >1 means data exists
        return len(rows) > 1
    except RuntimeError:
        return False


def drop_table(
    athena_client, database: str, workgroup: str, table_name: str, s3_output: Optional[str] = None
) -> None:
    """Drop a table from Athena."""
    exec_id = start_query(
        athena_client, f'DROP TABLE IF EXISTS `{table_name}`', database, workgroup, s3_output
    )
    result = wait_for_query(athena_client, exec_id)
    state = result["QueryExecution"]["Status"]["State"]
    if state != "SUCCEEDED":
        reason = result["QueryExecution"]["Status"].get("StateChangeReason", "unknown")
        print(f"  WARNING: DROP TABLE failed for '{table_name}': {reason}")


def drop_view(
    athena_client, database: str, workgroup: str, view_name: str, s3_output: Optional[str] = None
) -> None:
    """Drop a view from Athena."""
    exec_id = start_query(
        athena_client, f'DROP VIEW IF EXISTS `{view_name}`', database, workgroup, s3_output
    )
    result = wait_for_query(athena_client, exec_id)
    state = result["QueryExecution"]["Status"]["State"]
    if state != "SUCCEEDED":
        reason = result["QueryExecution"]["Status"].get("StateChangeReason", "unknown")
        print(f"  WARNING: DROP VIEW failed for '{view_name}': {reason}")


def aws_athena_cleanup(
    database: str,
    workgroup: str = "primary",
    region: str = "us-west-2",
    s3_output: Optional[str] = None,
    drop: bool = False,
    skip_views: bool = False,
) -> dict:
    """
    Scan an Athena database and identify (or drop) stale tables and views.

    A table is considered stale if:
      - Its S3 location no longer contains any objects, AND
      - A SELECT query returns zero rows.

    A view is considered stale if:
      - A SELECT query against it fails or returns zero rows.

    Parameters
    ----------
    database : str
        Athena/Glue database name.
    workgroup : str
        Athena workgroup.
    region : str
        AWS region.
    s3_output : str, optional
        S3 output location for query results.
    drop : bool
        If True, drop stale objects. If False, only report them.
    skip_views : bool
        If True, skip view inspection.

    Returns
    -------
    dict
        Summary with keys: 'stale_tables', 'stale_views', 'healthy_tables', 'healthy_views'.
    """
    athena_client, glue_client, s3_client = get_clients(region)

    summary = {
        "stale_tables": [],
        "stale_views": [],
        "healthy_tables": [],
        "healthy_views": [],
    }

    # --- Tables ---
    print(f"\n{'='*60}")
    print("AWS ATHENA DATABASE CLEANUP")
    print(f"{'-'*60}")
    print(f"Database:    {database}")
    print(f"Workgroup:   {workgroup}")
    print(f"Region:      {region}")
    print(f"S3 Output:   {s3_output or '(workgroup default)'}")
    print(f"Drop:        {drop}")
    print(f"Skip Views:  {skip_views}")
    print(f"{'='*60}")

    tables = list_tables(athena_client, database, workgroup, s3_output)
    views = list_views(athena_client, database, workgroup, s3_output) if not skip_views else []

    # Exclude views from table list (SHOW TABLES includes views in some Athena versions)
    tables_only = [t for t in tables if t not in views]

    print(f"\nFound {len(tables_only)} tables and {len(views)} views.\n")
    print("-" * 60)
    print("Checking tables...")
    print("-" * 60)

    for table in tables_only:
        s3_location = get_table_s3_location(glue_client, database, table)

        if s3_location:
            has_data = s3_path_has_data(s3_client, s3_location)
        else:
            # No S3 location means it might be a managed table or something else
            has_data = True  # don't flag as stale without evidence

        if not has_data:
            # Double-check: can we actually query any rows?
            has_rows = table_has_rows(athena_client, database, workgroup, table, s3_output)
            if not has_rows:
                print(f"  STALE: {table}")
                print(f"         S3: {s3_location} (empty/missing)")
                summary["stale_tables"].append(table)

                if drop:
                    print(f"         -> Dropping table '{table}'...")
                    drop_table(athena_client, database, workgroup, table, s3_output)
            else:
                print(f"  OK:    {table} (S3 empty but table returns rows — skipping)")
                summary["healthy_tables"].append(table)
        else:
            print(f"  OK:    {table}")
            summary["healthy_tables"].append(table)

    # --- Views ---
    if not skip_views:
        print()
        print("-" * 60)
        print("Checking views...")
        print("-" * 60)

        for view in views:
            has_rows = table_has_rows(athena_client, database, workgroup, view, s3_output)
            if not has_rows:
                print(f"  STALE: {view} (returns no rows or query failed)")
                summary["stale_views"].append(view)

                if drop:
                    print(f"         -> Dropping view '{view}'...")
                    drop_view(athena_client, database, workgroup, view, s3_output)
            else:
                print(f"  OK:    {view}")
                summary["healthy_views"].append(view)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'-'*60}")
    print(f"  Healthy tables: {len(summary['healthy_tables'])}")
    print(f"  Stale tables:   {len(summary['stale_tables'])}")
    print(f"  Healthy views:  {len(summary['healthy_views'])}")
    print(f"  Stale views:    {len(summary['stale_views'])}")
    if drop:
        print("Deleted all stale tables" + ("." if skip_views else " and views."))
    print(f"{'='*60}")
    if not drop and (summary["stale_tables"] or summary["stale_views"]):
        print("\n  Rerun with --drop (or -D) to remove stale objects.")
    print()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Scan an Athena database and remove tables/views with missing S3 data."
    )
    parser.add_argument("-d", "--database", required=True, help="Athena/Glue database name.")
    parser.add_argument("-w", "--workgroup", default="primary", help="Athena workgroup (default: primary).")
    parser.add_argument("-r", "--region", default="us-west-2", help="AWS region (default: us-west-2).")
    parser.add_argument("-o", "--s3-output", default=None,
                        help="S3 path for query results (if workgroup has no default).")
    parser.add_argument("-D", "--drop", action="store_true",
                        help="Actually drop stale tables/views. Without this flag, only reports.")
    parser.add_argument("-S", "--skip-views", action="store_true", help="Skip view inspection (only check tables).")
    args = parser.parse_args()

    aws_athena_cleanup(
        database=args.database,
        workgroup=args.workgroup,
        region=args.region,
        s3_output=args.s3_output,
        drop=args.drop,
        skip_views=args.skip_views,
    )


if __name__ == "__main__":
    main()
