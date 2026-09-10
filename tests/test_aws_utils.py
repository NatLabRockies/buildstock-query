"""Tests for buildstock_query.tools.aws_utils shared utilities."""

from unittest.mock import MagicMock, patch

import pytest

from buildstock_query.tools.aws_utils.athena import (
    get_workgroup_output_location,
    wait_for_query,
    start_query,
    run_query,
    list_tables,
    list_views,
)
from buildstock_query.tools.aws_utils.clients import get_clients, get_athena_client, get_glue_client, get_s3_client
from buildstock_query.tools.aws_utils.glue import list_databases, get_table_s3_location
from buildstock_query.tools.aws_utils.s3 import s3_path_has_data


class TestClients:
    @patch("buildstock_query.tools.aws_utils.clients.boto3")
    def test_get_clients_returns_tuple_of_three(self, mock_boto3):
        mock_boto3.client.return_value = MagicMock()
        athena, glue, s3 = get_clients("us-west-2")
        assert mock_boto3.client.call_count == 3

    @patch("buildstock_query.tools.aws_utils.clients.boto3")
    def test_get_athena_client(self, mock_boto3):
        get_athena_client("us-east-1")
        mock_boto3.client.assert_called_once_with("athena", region_name="us-east-1")

    @patch("buildstock_query.tools.aws_utils.clients.boto3")
    def test_get_glue_client(self, mock_boto3):
        get_glue_client("us-east-1")
        mock_boto3.client.assert_called_once_with("glue", region_name="us-east-1")

    @patch("buildstock_query.tools.aws_utils.clients.boto3")
    def test_get_s3_client(self, mock_boto3):
        get_s3_client("us-east-1")
        mock_boto3.client.assert_called_once_with("s3", region_name="us-east-1")


class TestGetWorkgroupOutputLocation:
    def test_returns_output_location(self):
        athena = MagicMock()
        athena.get_work_group.return_value = {
            "WorkGroup": {
                "Configuration": {
                    "ResultConfiguration": {
                        "OutputLocation": "s3://my-bucket/athena-results/"
                    }
                }
            }
        }
        loc = get_workgroup_output_location(athena, "my-workgroup")
        assert loc == "s3://my-bucket/athena-results/"

    def test_returns_none_when_no_config(self):
        athena = MagicMock()
        athena.get_work_group.return_value = {
            "WorkGroup": {"Configuration": {}}
        }
        loc = get_workgroup_output_location(athena, "my-workgroup")
        assert loc is None

    def test_returns_none_on_client_error(self):
        from botocore.exceptions import ClientError

        athena = MagicMock()
        error_response = {"Error": {"Code": "InvalidRequestException", "Message": "not found"}}
        athena.get_work_group.side_effect = ClientError(error_response, "GetWorkGroup")
        loc = get_workgroup_output_location(athena, "bad-workgroup")
        assert loc is None


class TestWaitForQuery:
    def test_returns_immediately_on_succeeded(self):
        athena = MagicMock()
        athena.get_query_execution.return_value = {
            "QueryExecution": {"Status": {"State": "SUCCEEDED"}}
        }
        result = wait_for_query(athena, "exec-123", max_wait=10)
        assert result["QueryExecution"]["Status"]["State"] == "SUCCEEDED"

    def test_returns_on_failed(self):
        athena = MagicMock()
        athena.get_query_execution.return_value = {
            "QueryExecution": {"Status": {"State": "FAILED", "StateChangeReason": "bad sql"}}
        }
        result = wait_for_query(athena, "exec-123", max_wait=10)
        assert result["QueryExecution"]["Status"]["State"] == "FAILED"

    def test_returns_on_cancelled(self):
        athena = MagicMock()
        athena.get_query_execution.return_value = {
            "QueryExecution": {"Status": {"State": "CANCELLED"}}
        }
        result = wait_for_query(athena, "exec-123", max_wait=10)
        assert result["QueryExecution"]["Status"]["State"] == "CANCELLED"

    @patch("buildstock_query.tools.aws_utils.athena.time.sleep")
    def test_timeout_raises(self, mock_sleep):
        athena = MagicMock()
        athena.get_query_execution.return_value = {
            "QueryExecution": {"Status": {"State": "RUNNING"}}
        }
        with pytest.raises(TimeoutError, match="did not complete"):
            wait_for_query(athena, "exec-123", max_wait=4)


class TestStartQuery:
    def test_returns_execution_id(self):
        athena = MagicMock()
        athena.start_query_execution.return_value = {"QueryExecutionId": "abc-123"}
        exec_id = start_query(athena, "SELECT 1", "mydb", "primary")
        assert exec_id == "abc-123"

    def test_with_s3_output(self):
        athena = MagicMock()
        athena.start_query_execution.return_value = {"QueryExecutionId": "abc-123"}
        start_query(athena, "SELECT 1", "mydb", "primary", s3_output="s3://bucket/path/")
        call_kwargs = athena.start_query_execution.call_args[1]
        assert call_kwargs["ResultConfiguration"] == {"OutputLocation": "s3://bucket/path/"}

    def test_missing_output_location_raises_runtime_error(self):
        from botocore.exceptions import ClientError

        athena = MagicMock()
        error_response = {"Error": {"Code": "InvalidRequestException", "Message": "No output location"}}
        athena.start_query_execution.side_effect = ClientError(error_response, "StartQueryExecution")

        with pytest.raises(RuntimeError, match="no default output location"):
            start_query(athena, "SELECT 1", "mydb", "primary")

    def test_other_client_error_reraises(self):
        from botocore.exceptions import ClientError

        athena = MagicMock()
        error_response = {"Error": {"Code": "AccessDeniedException", "Message": "Not authorized"}}
        athena.start_query_execution.side_effect = ClientError(error_response, "StartQueryExecution")

        with pytest.raises(ClientError):
            start_query(athena, "SELECT 1", "mydb", "primary")


class TestRunQuery:
    @patch("buildstock_query.tools.aws_utils.athena.wait_for_query")
    def test_returns_rows_on_success(self, mock_wait):
        athena = MagicMock()
        athena.start_query_execution.return_value = {"QueryExecutionId": "exec-1"}
        mock_wait.return_value = {
            "QueryExecution": {"Status": {"State": "SUCCEEDED"}}
        }
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"ResultSet": {"Rows": [
                {"Data": [{"VarCharValue": "col1"}]},
                {"Data": [{"VarCharValue": "val1"}]},
            ]}}
        ]
        athena.get_paginator.return_value = paginator

        rows = run_query(athena, "SELECT 1", "mydb", "primary")
        assert len(rows) == 2

    @patch("buildstock_query.tools.aws_utils.athena.wait_for_query")
    def test_raises_on_failed_query(self, mock_wait):
        athena = MagicMock()
        athena.start_query_execution.return_value = {"QueryExecutionId": "exec-1"}
        mock_wait.return_value = {
            "QueryExecution": {"Status": {"State": "FAILED", "StateChangeReason": "syntax error"}}
        }

        with pytest.raises(RuntimeError, match="Query failed"):
            run_query(athena, "BAD SQL", "mydb", "primary")


class TestListTables:
    @patch("buildstock_query.tools.aws_utils.athena.run_query")
    def test_returns_table_names(self, mock_run):
        mock_run.return_value = [
            {"Data": [{"VarCharValue": "table_a"}]},
            {"Data": [{"VarCharValue": "table_b"}]},
        ]
        tables = list_tables(MagicMock(), "mydb", "primary")
        assert tables == ["table_a", "table_b"]


class TestListViews:
    @patch("buildstock_query.tools.aws_utils.athena.run_query")
    def test_returns_view_names(self, mock_run):
        mock_run.return_value = [
            {"Data": [{"VarCharValue": "view_x"}]},
        ]
        views = list_views(MagicMock(), "mydb", "primary")
        assert views == ["view_x"]


class TestListDatabases:
    def test_paginates_and_returns_names(self):
        glue = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"DatabaseList": [{"Name": "db1"}, {"Name": "db2"}]},
            {"DatabaseList": [{"Name": "db3"}]},
        ]
        glue.get_paginator.return_value = paginator

        dbs = list_databases(glue)
        assert dbs == ["db1", "db2", "db3"]


class TestGetTableS3Location:
    def test_returns_location(self):
        glue = MagicMock()
        glue.get_table.return_value = {
            "Table": {"StorageDescriptor": {"Location": "s3://bucket/prefix/"}}
        }
        loc = get_table_s3_location(glue, "mydb", "my_table")
        assert loc == "s3://bucket/prefix/"

    def test_returns_none_for_entity_not_found(self):
        from botocore.exceptions import ClientError

        glue = MagicMock()
        error_response = {"Error": {"Code": "EntityNotFoundException", "Message": "not found"}}
        glue.get_table.side_effect = ClientError(error_response, "GetTable")

        loc = get_table_s3_location(glue, "mydb", "missing_table")
        assert loc is None

    def test_reraises_other_errors(self):
        from botocore.exceptions import ClientError

        glue = MagicMock()
        error_response = {"Error": {"Code": "AccessDeniedException", "Message": "denied"}}
        glue.get_table.side_effect = ClientError(error_response, "GetTable")

        with pytest.raises(ClientError):
            get_table_s3_location(glue, "mydb", "my_table")


class TestS3PathHasData:
    def test_returns_true_when_objects_exist(self):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {"KeyCount": 1}
        assert s3_path_has_data(s3, "s3://bucket/prefix/") is True

    def test_returns_false_when_empty(self):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {"KeyCount": 0}
        assert s3_path_has_data(s3, "s3://bucket/prefix/") is False

    def test_returns_false_for_invalid_uri(self):
        s3 = MagicMock()
        assert s3_path_has_data(s3, "") is False
        assert s3_path_has_data(s3, "http://not-s3/path") is False
        assert s3_path_has_data(s3, None) is False

    def test_returns_false_on_client_error(self):
        from botocore.exceptions import ClientError

        s3 = MagicMock()
        error_response = {"Error": {"Code": "NoSuchBucket", "Message": "not found"}}
        s3.list_objects_v2.side_effect = ClientError(error_response, "ListObjectsV2")
        assert s3_path_has_data(s3, "s3://nonexistent/path/") is False

    def test_appends_slash_to_prefix(self):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {"KeyCount": 1}
        s3_path_has_data(s3, "s3://bucket/prefix")
        call_kwargs = s3.list_objects_v2.call_args[1]
        assert call_kwargs["Prefix"] == "prefix/"
