from __future__ import annotations

import datetime
import logging
import hashlib
from unittest.mock import MagicMock

import pandas as pd

from buildstock_query.query_core import QueryCore


def _make_query_core(pages: list[dict]) -> tuple[QueryCore, MagicMock]:
    qc = QueryCore.__new__(QueryCore)
    paginator = MagicMock()
    paginator.paginate.return_value = pages
    qc._aws_s3 = MagicMock()
    qc._aws_s3.get_paginator.return_value = paginator
    return qc, paginator


class TestQueryCoreCachePaths:
    def test_get_query_result_location_selects_newest_leaf_folder(self, caplog) -> None:
        base = "bsq_athena_unload_results/abc123/"
        pages = [{
            "Contents": [
                {
                    "Key": f"{base}older/part-0.parquet",
                    "LastModified": datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
                },
                {
                    "Key": f"{base}newer/part-0.parquet",
                    "LastModified": datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc),
                },
                {
                    "Key": f"{base}newer/part-1.parquet",
                    "LastModified": datetime.datetime(2024, 1, 3, tzinfo=datetime.timezone.utc),
                },
            ]
        }]
        qc, paginator = _make_query_core(pages)

        with caplog.at_level(logging.WARNING):
            result = qc._get_query_result_location("s3://test-bucket/bsq_athena_unload_results/abc123")

        assert result == "s3://test-bucket/bsq_athena_unload_results/abc123/newer/"
        paginator.paginate.assert_called_once_with(
            Bucket="test-bucket",
            Prefix="bsq_athena_unload_results/abc123/",
        )
        assert "Multiple cached UNLOAD result folders found" in caplog.text

    def test_get_query_result_location_returns_none_without_child_folders(self) -> None:
        pages = [{
            "Contents": [
                {
                    "Key": "bsq_athena_unload_results/abc123/_SUCCESS",
                    "LastModified": datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
                },
                {
                    "Key": "bsq_athena_unload_results/abc123/",
                    "LastModified": datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
                },
            ]
        }]
        qc, _ = _make_query_core(pages)

        result = qc._get_query_result_location("s3://test-bucket/bsq_athena_unload_results/abc123/")

        assert result is None

    def test_get_query_result_location_ignores_root_level_keys(self) -> None:
        base = "bsq_athena_unload_results/abc123/"
        pages = [{
            "Contents": [
                {
                    "Key": f"{base}_SUCCESS",
                    "LastModified": datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
                },
                {
                    "Key": f"{base}leaf/part-0.parquet",
                    "LastModified": datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc),
                },
            ]
        }]
        qc, _ = _make_query_core(pages)

        result = qc._get_query_result_location("s3://test-bucket/bsq_athena_unload_results/abc123")

        assert result == "s3://test-bucket/bsq_athena_unload_results/abc123/leaf/"

    def test_execute_keeps_unload_namespace_for_prefixed_s3_root(self) -> None:
        query = "SELECT 1"
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        expected_root = (
            f"s3://test-bucket/custom/prefix/bsq_athena_unload_results/{query_hash}"
        )
        cached_location = f"{expected_root}/cached/"
        expected_df = pd.DataFrame({"value": [1]})

        qc = QueryCore.__new__(QueryCore)
        qc.run_params = MagicMock(
            query_unload_s3_bucket="test-bucket/custom/prefix",
            athena_query_reuse=True,
        )
        qc._session_queries = set()
        qc._query_cache = {}
        qc._get_query_result_location = MagicMock(return_value=cached_location)
        qc._read_s3_parquet = MagicMock(return_value=expected_df)

        result = qc.execute(query)

        qc._get_query_result_location.assert_called_once_with(expected_root)
        qc._read_s3_parquet.assert_called_once_with(cached_location)
        pd.testing.assert_frame_equal(result, expected_df)
