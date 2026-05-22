"""Boto3 client factory functions."""

import boto3


def get_athena_client(region_name: str = "us-west-2"):
    """Create a boto3 Athena client."""
    return boto3.client("athena", region_name=region_name)


def get_glue_client(region_name: str = "us-west-2"):
    """Create a boto3 Glue client."""
    return boto3.client("glue", region_name=region_name)


def get_s3_client(region_name: str = "us-west-2"):
    """Create a boto3 S3 client."""
    return boto3.client("s3", region_name=region_name)


def get_clients(region_name: str = "us-west-2"):
    """Create boto3 Athena, Glue, and S3 clients."""
    return (
        get_athena_client(region_name),
        get_glue_client(region_name),
        get_s3_client(region_name),
    )
