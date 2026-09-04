"""AWS Glue catalog helpers."""

from typing import Optional

from botocore.exceptions import ClientError


def list_databases(glue_client) -> list[str]:
    """List all databases in the Glue catalog."""
    databases = []
    paginator = glue_client.get_paginator("get_databases")
    for page in paginator.paginate():
        for db in page["DatabaseList"]:
            databases.append(db["Name"])
    return databases


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
