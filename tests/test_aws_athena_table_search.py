"""Tests for buildstock_query.tools.aws_athena_table_search."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from buildstock_query.tools.aws_athena_table_search import (
    aws_athena_table_search,
    search_table_in_database,
)


def _make_glue_table(name, location="s3://bucket/prefix/", create_time=None):
    """Helper to create a mock Glue table entry."""
    return {
        "Name": name,
        "StorageDescriptor": {"Location": location},
        "CreateTime": create_time or datetime(2024, 1, 15),
    }


class TestSearchTableInDatabase:
    def test_exact_match(self):
        glue = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"TableList": [
                _make_glue_table("baseline"),
                _make_glue_table("timeseries"),
                _make_glue_table("upgrades"),
            ]}
        ]
        glue.get_paginator.return_value = paginator

        matches = search_table_in_database(glue, "mydb", "baseline")
        assert len(matches) == 1
        assert matches[0]["table"] == "baseline"
        assert matches[0]["database"] == "mydb"

    def test_exact_match_case_insensitive(self):
        glue = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"TableList": [_make_glue_table("Baseline_2024")]}
        ]
        glue.get_paginator.return_value = paginator

        matches = search_table_in_database(glue, "mydb", "baseline_2024")
        assert len(matches) == 1

    def test_substring_match(self):
        glue = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"TableList": [
                _make_glue_table("res_baseline_v1"),
                _make_glue_table("com_baseline_v2"),
                _make_glue_table("timeseries"),
            ]}
        ]
        glue.get_paginator.return_value = paginator

        matches = search_table_in_database(glue, "mydb", "baseline", substring=True)
        assert len(matches) == 2
        assert all("baseline" in m["table"] for m in matches)

    def test_regex_match(self):
        glue = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"TableList": [
                _make_glue_table("baseline_10"),
                _make_glue_table("baseline_25"),
                _make_glue_table("baseline_summary"),
            ]}
        ]
        glue.get_paginator.return_value = paginator

        matches = search_table_in_database(glue, "mydb", r"baseline_\d+", regex=True)
        assert len(matches) == 2
        assert {m["table"] for m in matches} == {"baseline_10", "baseline_25"}

    def test_no_matches_returns_empty(self):
        glue = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"TableList": [_make_glue_table("unrelated_table")]}
        ]
        glue.get_paginator.return_value = paginator

        matches = search_table_in_database(glue, "mydb", "nonexistent")
        assert matches == []

    def test_access_denied_skipped(self, capsys):
        glue = MagicMock()
        paginator = MagicMock()
        error_response = {"Error": {"Code": "AccessDeniedException", "Message": "denied"}}
        paginator.paginate.side_effect = ClientError(error_response, "GetTables")
        glue.get_paginator.return_value = paginator

        matches = search_table_in_database(glue, "restricted_db", "table")
        assert matches == []
        captured = capsys.readouterr()
        assert "Access denied" in captured.out

    def test_other_error_reraises(self):
        glue = MagicMock()
        paginator = MagicMock()
        error_response = {"Error": {"Code": "InternalServiceException", "Message": "boom"}}
        paginator.paginate.side_effect = ClientError(error_response, "GetTables")
        glue.get_paginator.return_value = paginator

        with pytest.raises(ClientError):
            search_table_in_database(glue, "mydb", "table")

    def test_table_without_storage_descriptor(self):
        glue = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"TableList": [{"Name": "view_table", "CreateTime": None}]}
        ]
        glue.get_paginator.return_value = paginator

        matches = search_table_in_database(glue, "mydb", "view_table")
        assert len(matches) == 1
        assert matches[0]["s3_location"] == "(none)"
        assert matches[0]["create_time"] == "(unknown)"


@patch("buildstock_query.tools.aws_athena_table_search.list_databases")
@patch("buildstock_query.tools.aws_athena_table_search.get_glue_client")
class TestAwsAthenaTableSearch:
    def test_searches_all_databases(self, mock_client, mock_list_dbs):
        mock_client.return_value = MagicMock()
        mock_list_dbs.return_value = ["db1", "db2", "db3"]

        with patch(
            "buildstock_query.tools.aws_athena_table_search.search_table_in_database"
        ) as mock_search:
            mock_search.side_effect = [
                [{"database": "db1", "table": "target", "s3_location": "s3://b/p/", "create_time": "2024-01-01"}],
                [],
                [{"database": "db3", "table": "target", "s3_location": "s3://b/q/", "create_time": "2024-02-01"}],
            ]

            matches = aws_athena_table_search(table_name="target", region="us-west-2")
            assert len(matches) == 2
            assert mock_search.call_count == 3

    def test_database_filter(self, mock_client, mock_list_dbs):
        mock_client.return_value = MagicMock()
        mock_list_dbs.return_value = ["resstock_core", "comstock_v1", "resstock_test"]

        with patch(
            "buildstock_query.tools.aws_athena_table_search.search_table_in_database"
        ) as mock_search:
            mock_search.return_value = []

            aws_athena_table_search(table_name="baseline", region="us-west-2", database_filter="resstock")
            # Should only search resstock_core and resstock_test
            assert mock_search.call_count == 2
            searched_dbs = [call[0][1] for call in mock_search.call_args_list]
            assert "comstock_v1" not in searched_dbs

    def test_no_matches_returns_empty_list(self, mock_client, mock_list_dbs):
        mock_client.return_value = MagicMock()
        mock_list_dbs.return_value = ["db1"]

        with patch(
            "buildstock_query.tools.aws_athena_table_search.search_table_in_database"
        ) as mock_search:
            mock_search.return_value = []

            matches = aws_athena_table_search(table_name="nonexistent", region="us-west-2")
            assert matches == []

    def test_passes_substring_flag(self, mock_client, mock_list_dbs):
        mock_client.return_value = MagicMock()
        mock_list_dbs.return_value = ["db1"]

        with patch(
            "buildstock_query.tools.aws_athena_table_search.search_table_in_database"
        ) as mock_search:
            mock_search.return_value = []

            aws_athena_table_search(table_name="base", region="us-west-2", substring=True)
            call_kwargs = mock_search.call_args
            assert call_kwargs[0][2] == "base"  # table_name
            assert call_kwargs[0][3] is True  # substring
            assert call_kwargs[0][4] is False  # regex

    def test_passes_regex_flag(self, mock_client, mock_list_dbs):
        mock_client.return_value = MagicMock()
        mock_list_dbs.return_value = ["db1"]

        with patch(
            "buildstock_query.tools.aws_athena_table_search.search_table_in_database"
        ) as mock_search:
            mock_search.return_value = []

            aws_athena_table_search(table_name=r"base_\d+", region="us-west-2", regex=True)
            call_kwargs = mock_search.call_args
            assert call_kwargs[0][4] is True  # regex
