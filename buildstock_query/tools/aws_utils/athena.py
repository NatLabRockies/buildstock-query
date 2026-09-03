"""Athena query execution helpers."""

import time
from typing import Optional

from botocore.exceptions import ClientError


def get_workgroup_output_location(athena_client, workgroup: str) -> Optional[str]:
    """Get the default output location configured for an Athena workgroup."""
    try:
        response = athena_client.get_work_group(WorkGroup=workgroup)
        config = response["WorkGroup"].get("Configuration", {})
        return config.get("ResultConfiguration", {}).get("OutputLocation")
    except ClientError:
        return None


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
                f"Configure the workgroup's result output location in AWS."
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
    rows = run_query(athena_client, "SHOW TABLES", database, workgroup, s3_output)
    return [row["Data"][0]["VarCharValue"] for row in rows]


def list_views(athena_client, database: str, workgroup: str, s3_output: Optional[str] = None) -> list[str]:
    """List all views in the database."""
    rows = run_query(athena_client, "SHOW VIEWS", database, workgroup, s3_output)
    return [row["Data"][0]["VarCharValue"] for row in rows]
