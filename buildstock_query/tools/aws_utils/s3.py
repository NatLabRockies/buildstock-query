"""AWS S3 helpers."""

from botocore.exceptions import ClientError


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
