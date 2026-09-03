"""Tests for QueryCore S3 parquet read path using moto.

These tests exercise QueryCore._read_s3_parquet end-to-end against a
moto-backed S3 endpoint, without using ``unittest.mock``. They cover:
  * the ``s3://`` prefix stripping and PyArrow filesystem wiring
  * a successful round-trip through ``pd.read_parquet`` with the
    provided PyArrow filesystem
  * the failure -> refresh -> retry control flow in ``_read_s3_parquet``
"""

from __future__ import annotations

import socket

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.fs as _pafs
import pyarrow.parquet as pq
import pytest
from moto.server import ThreadedMotoServer

from buildstock_query.query_core import QueryCore


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _RunParamsStub:
    """Minimal stand-in for ``RunParams`` exposing only ``region_name``."""

    def __init__(self, region_name: str) -> None:
        self.region_name = region_name


@pytest.fixture
def moto_s3():
    """Start a real moto S3 server and configure boto3 env credentials.

    Yields a tuple of (endpoint_url, region).
    """
    port = _free_port()
    server = ThreadedMotoServer(port=port)
    server.start()
    endpoint = f"http://127.0.0.1:{port}"
    region = "us-east-1"

    monkey_env = {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": region,
    }
    import os

    saved = {k: os.environ.get(k) for k in monkey_env}
    os.environ.update(monkey_env)
    try:
        yield endpoint, region
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        server.stop()


def _make_s3fs(endpoint: str, region: str, **overrides) -> _pafs.S3FileSystem:
    kwargs = dict(
        access_key="testing",
        secret_key="testing",
        session_token="testing",
        region=region,
        endpoint_override=endpoint,
        scheme="http",
    )
    kwargs.update(overrides)
    return _pafs.S3FileSystem(**kwargs)


def _put_parquet(
    s3_client, bucket: str, key: str, df: pd.DataFrame
) -> None:
    """Write ``df`` as parquet to ``s3://bucket/key`` using boto3.

    Uses an in-memory parquet buffer + ``put_object`` rather than PyArrow's
    multipart-upload code path, which doesn't round-trip cleanly against
    moto's S3 implementation.
    """
    import io

    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


class TestReadS3Parquet:
    def test_read_s3_parquet_roundtrip_via_moto(self, moto_s3) -> None:
        endpoint, region = moto_s3
        bucket = "bsq-test-bucket"
        key = "data/foo.parquet"

        boto3.client("s3", endpoint_url=endpoint, region_name=region).create_bucket(Bucket=bucket)
        fs = _make_s3fs(endpoint, region)
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        _put_parquet(
            boto3.client("s3", endpoint_url=endpoint, region_name=region),
            bucket, key, df,
        )

        qc = QueryCore.__new__(QueryCore)
        qc.run_params = _RunParamsStub(region)
        qc._pa_s3fs = fs

        result = qc._read_s3_parquet(f"s3://{bucket}/{key}")

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True).sort_index(axis=1),
            df.sort_index(axis=1),
        )

    def test_read_s3_parquet_rejects_non_s3_path(self) -> None:
        qc = QueryCore.__new__(QueryCore)
        qc.run_params = _RunParamsStub("us-east-1")
        qc._pa_s3fs = _pafs.LocalFileSystem()

        with pytest.raises(ValueError, match="Expected an s3:// path"):
            qc._read_s3_parquet("not-s3://bucket/key")

    def test_read_s3_parquet_refreshes_filesystem_on_failure(self, moto_s3) -> None:
        endpoint, region = moto_s3
        bucket = "bsq-test-bucket-retry"
        key = "data/bar.parquet"

        boto3.client("s3", endpoint_url=endpoint, region_name=region).create_bucket(Bucket=bucket)
        good_fs = _make_s3fs(endpoint, region)
        df = pd.DataFrame({"x": [10, 20, 30]})
        _put_parquet(
            boto3.client("s3", endpoint_url=endpoint, region_name=region),
            bucket, key, df,
        )

        # A "stale" filesystem pointing at an unreachable endpoint to force the
        # first read to fail and trigger the refresh+retry path.
        bad_fs = _make_s3fs(
            "http://127.0.0.1:1",
            region,
            access_key="bad",
            secret_key="bad",
            session_token=None,
        )

        qc = QueryCore.__new__(QueryCore)
        qc.run_params = _RunParamsStub(region)
        qc._pa_s3fs = bad_fs
        # Stand-in for the credential refresh: replace with the working filesystem.
        qc._create_pa_s3_filesystem = lambda: good_fs  # type: ignore[method-assign]

        result = qc._read_s3_parquet(f"s3://{bucket}/{key}")

        assert qc._pa_s3fs is good_fs
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True).sort_index(axis=1),
            df.sort_index(axis=1),
        )
