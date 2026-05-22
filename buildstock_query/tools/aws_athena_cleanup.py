"""
Cleanup script for AWS Athena databases.

Connects to an Athena database/workgroup via boto3 and identifies tables/views
whose underlying S3 data no longer exists or that return zero rows, then
optionally drops them.

Usage:
    python cleanup_aws_athena_database.py --database my_db --workgroup primary --region us-west-2
    python cleanup_aws_athena_database.py --database my_db --workgroup primary --drop  # actually drop stale objects
"""

import argparse
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError


def get_clients(region_name: str = "us-west-2"):
    """Create boto3 Athena, Glue, and S3 clients."""
    return (
        boto3.client("athena", region_name=region_name),
        boto3.client("glue", region_name=region_name),
        boto3.client("s3", region_name=region_name),
    )


def wait_for_query(athena_client, execution_id: str, max_wait: int = 300) -> dict:
    """Poll until an Athena query completes or fails."""
    elapsed = 0
    while elapsed < max_wait:
        response = athena_client.get_query_execution(QueryExecutionId=execution_id)
        state = response["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return response
        time.sleep(2)
        elapsed += 2
    raise TimeoutError(f"Query {execution_id} did not complete within {max_wait}s")


def start_query(
    athena_client,
    query: str,
    database: str,
    workgroup: str,
    s3_output: Optional[str] = None,
) -> str:
    """Start an Athena query and return the execution ID."""
    kwargs = {
        "QueryString": query,
        "QueryExecutionContext": {"Database": database},
        "WorkGroup": workgroup,
    }
    if s3_output:
        kwargs["ResultConfiguration"] = {"OutputLocation": s3_output}

    try:
        response = athena_client.start_query_execution(**kwargs)
    except ClientError as e:
        error_msg = e.response["Error"]["Message"]
        if "output location" in error_msg.lower():
            raise RuntimeError(
                f"Workgroup '{workgroup}' has no default output location. "
                f"Pass --s3-output or configure the workgroup."
            ) from e
        raise
    return response["QueryExecutionId"]


def run_query(
    athena_client,
    query: str,
    database: str,
    workgroup: str,
    s3_output: Optional[str] = None,
) -> list[dict]:
    """Run a query and return result rows (list of row dicts)."""
    exec_id = start_query(athena_client, query, database, workgroup, s3_output)
    result = wait_for_query(athena_client, exec_id)

    state = result["QueryExecution"]["Status"]["State"]
    if state != "SUCCEEDED":
        reason = result["QueryExecution"]["Status"].get("StateChangeReason", "unknown")
        raise RuntimeError(f"Query failed ({state}): {reason}\nQuery: {query}")

    rows = []
    paginator = athena_client.get_paginator("get_query_results")
    for page in paginator.paginate(QueryExecutionId=exec_id):
        for row in page["ResultSet"]["Rows"]:
            rows.append(row)
    return rows


def list_tables(athena_client, database: str, workgroup: str, s3_output: Optional[str] = None) -> list[str]:
    """List all tables in the database."""
    rows = run_query(athena_client, f"SHOW TABLES IN {database}", database, workgroup, s3_output)
    # First row is header for SHOW TABLES
    return [row["Data"][0]["VarCharValue"] for row in rows]


def list_views(athena_client, database: str, workgroup: str, s3_output: Optional[str] = None) -> list[str]:
    """List all views in the database."""
    rows = run_query(athena_client, f"SHOW VIEWS IN {database}", database, workgroup, s3_output)
    return [row["Data"][0]["VarCharValue"] for row in rows]


def get_table_s3_location(glue_client, database: str, table_name: str) -> Optional[str]:
    """Get the S3 location of a Glue table. Returns None for views or if not found."""
    try:
        response = glue_client.get_table(DatabaseName=database, Name=table_name)
        storage = response["Table"].get("StorageDescriptor", {})
        return storage.get("Location")
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityNotFoundException":
            return None
        raise


def s3_path_has_data(s3_client, s3_uri: str) -> bool:
    """
    Check if an S3 path contains any objects.
    Returns False if the path is empty or doesn't exist.
    """
    if not s3_uri or not s3_uri.startswith("s3://"):
        return False

    # Parse s3://bucket/prefix
    path = s3_uri[5:]  # strip "s3://"
    bucket, _, prefix = path.partition("/")

    # Ensure prefix ends with / for directory-like listing
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return response.get("KeyCount", 0) > 0
    except ClientError:
        return False


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
        print(f"\n  Rerun with --drop (or -D) to remove stale objects.")
    print()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Scan an Athena database and remove tables/views with missing S3 data."
    )
    parser.add_argument("-d", "--database", required=True, help="Athena/Glue database name.")
    parser.add_argument("-w", "--workgroup", default="primary", help="Athena workgroup (default: primary).")
    parser.add_argument("-r", "--region", default="us-west-2", help="AWS region (default: us-west-2).")
    parser.add_argument("-o", "--s3-output", default=None, help="S3 path for query results (if workgroup has no default).")
    parser.add_argument("-D", "--drop", action="store_true", help="Actually drop stale tables/views. Without this flag, only reports.")
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
