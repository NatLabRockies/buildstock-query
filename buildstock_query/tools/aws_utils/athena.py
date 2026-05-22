"""Athena query execution helpers."""

import time
from typing import Optional

from botocore.exceptions import ClientError


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
    return [row["Data"][0]["VarCharValue"] for row in rows]


def list_views(athena_client, database: str, workgroup: str, s3_output: Optional[str] = None) -> list[str]:
    """List all views in the database."""
    rows = run_query(athena_client, f"SHOW VIEWS IN {database}", database, workgroup, s3_output)
    return [row["Data"][0]["VarCharValue"] for row in rows]
