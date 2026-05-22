"""Tests for buildstock_query.tools.aws_athena_cleanup."""

from unittest.mock import MagicMock, patch

from buildstock_query.tools.aws_athena_cleanup import (
    aws_athena_cleanup,
    table_has_rows,
    drop_table,
    drop_view,
)


@patch("buildstock_query.tools.aws_athena_cleanup.run_query")
class TestTableHasRows:
    def test_returns_true_when_data_present(self, mock_run):
        # Header row + 1 data row = has rows
        mock_run.return_value = [
            {"Data": [{"VarCharValue": "1"}]},
            {"Data": [{"VarCharValue": "1"}]},
        ]
        assert table_has_rows(MagicMock(), "db", "wg", "my_table") is True

    def test_returns_false_when_only_header(self, mock_run):
        # Only header row = no data
        mock_run.return_value = [
            {"Data": [{"VarCharValue": "1"}]},
        ]
        assert table_has_rows(MagicMock(), "db", "wg", "my_table") is False

    def test_returns_false_on_runtime_error(self, mock_run):
        mock_run.side_effect = RuntimeError("Query failed")
        assert table_has_rows(MagicMock(), "db", "wg", "my_table") is False


@patch("buildstock_query.tools.aws_athena_cleanup.wait_for_query")
@patch("buildstock_query.tools.aws_athena_cleanup.start_query")
class TestDropTable:
    def test_drop_table_success(self, mock_start, mock_wait):
        mock_start.return_value = "exec-1"
        mock_wait.return_value = {
            "QueryExecution": {"Status": {"State": "SUCCEEDED"}}
        }
        # Should not raise
        drop_table(MagicMock(), "db", "wg", "stale_table")
        mock_start.assert_called_once()
        assert "DROP TABLE" in mock_start.call_args[0][1]

    def test_drop_table_failure_prints_warning(self, mock_start, mock_wait, capsys):
        mock_start.return_value = "exec-1"
        mock_wait.return_value = {
            "QueryExecution": {"Status": {"State": "FAILED", "StateChangeReason": "access denied"}}
        }
        drop_table(MagicMock(), "db", "wg", "stale_table")
        captured = capsys.readouterr()
        assert "WARNING" in captured.out


@patch("buildstock_query.tools.aws_athena_cleanup.wait_for_query")
@patch("buildstock_query.tools.aws_athena_cleanup.start_query")
class TestDropView:
    def test_drop_view_success(self, mock_start, mock_wait):
        mock_start.return_value = "exec-1"
        mock_wait.return_value = {
            "QueryExecution": {"Status": {"State": "SUCCEEDED"}}
        }
        drop_view(MagicMock(), "db", "wg", "stale_view")
        assert "DROP VIEW" in mock_start.call_args[0][1]

    def test_drop_view_failure_prints_warning(self, mock_start, mock_wait, capsys):
        mock_start.return_value = "exec-1"
        mock_wait.return_value = {
            "QueryExecution": {"Status": {"State": "FAILED", "StateChangeReason": "error"}}
        }
        drop_view(MagicMock(), "db", "wg", "stale_view")
        captured = capsys.readouterr()
        assert "WARNING" in captured.out


@patch("buildstock_query.tools.aws_athena_cleanup.s3_path_has_data")
@patch("buildstock_query.tools.aws_athena_cleanup.get_table_s3_location")
@patch("buildstock_query.tools.aws_athena_cleanup.list_views")
@patch("buildstock_query.tools.aws_athena_cleanup.list_tables")
@patch("buildstock_query.tools.aws_athena_cleanup.get_clients")
class TestAwsAthenaCleanup:
    def test_all_healthy(self, mock_clients, mock_tables, mock_views, mock_location, mock_s3):
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_tables.return_value = ["table_a", "table_b"]
        mock_views.return_value = []
        mock_location.return_value = "s3://bucket/table_a/"
        mock_s3.return_value = True  # S3 has data

        summary = aws_athena_cleanup(database="db", workgroup="wg", region="us-west-2")
        assert summary["stale_tables"] == []
        assert "table_a" in summary["healthy_tables"]
        assert "table_b" in summary["healthy_tables"]

    @patch("buildstock_query.tools.aws_athena_cleanup.table_has_rows")
    def test_identifies_stale_table(self, mock_rows, mock_clients, mock_tables, mock_views, mock_location, mock_s3):
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_tables.return_value = ["good_table", "stale_table"]
        mock_views.return_value = []
        mock_location.side_effect = lambda g, d, t: "s3://bucket/good/" if t == "good_table" else "s3://bucket/stale/"
        mock_s3.side_effect = lambda s3, uri: "good" in uri
        mock_rows.return_value = False  # stale table has no rows

        summary = aws_athena_cleanup(database="db", workgroup="wg", region="us-west-2")
        assert "stale_table" in summary["stale_tables"]
        assert "good_table" in summary["healthy_tables"]

    @patch("buildstock_query.tools.aws_athena_cleanup.table_has_rows")
    def test_s3_empty_but_has_rows_is_healthy(self, mock_rows, mock_clients, mock_tables, mock_views, mock_location, mock_s3):
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_tables.return_value = ["weird_table"]
        mock_views.return_value = []
        mock_location.return_value = "s3://bucket/weird/"
        mock_s3.return_value = False  # S3 is empty
        mock_rows.return_value = True  # but table has rows

        summary = aws_athena_cleanup(database="db", workgroup="wg", region="us-west-2")
        assert "weird_table" in summary["healthy_tables"]
        assert summary["stale_tables"] == []

    def test_no_s3_location_treated_as_healthy(self, mock_clients, mock_tables, mock_views, mock_location, mock_s3):
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_tables.return_value = ["managed_table"]
        mock_views.return_value = []
        mock_location.return_value = None  # no S3 location
        mock_s3.return_value = False

        summary = aws_athena_cleanup(database="db", workgroup="wg", region="us-west-2")
        assert "managed_table" in summary["healthy_tables"]

    @patch("buildstock_query.tools.aws_athena_cleanup.drop_table")
    @patch("buildstock_query.tools.aws_athena_cleanup.table_has_rows")
    def test_drop_mode_calls_drop_table(self, mock_rows, mock_drop, mock_clients, mock_tables, mock_views, mock_location, mock_s3):
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_tables.return_value = ["stale_table"]
        mock_views.return_value = []
        mock_location.return_value = "s3://bucket/stale/"
        mock_s3.return_value = False
        mock_rows.return_value = False

        aws_athena_cleanup(database="db", workgroup="wg", region="us-west-2", drop=True)
        mock_drop.assert_called_once()

    def test_skip_views(self, mock_clients, mock_tables, mock_views, mock_location, mock_s3):
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_tables.return_value = ["table_a"]
        mock_location.return_value = "s3://bucket/a/"
        mock_s3.return_value = True

        summary = aws_athena_cleanup(database="db", workgroup="wg", region="us-west-2", skip_views=True)
        mock_views.assert_not_called()
        assert summary["healthy_views"] == []
        assert summary["stale_views"] == []

    @patch("buildstock_query.tools.aws_athena_cleanup.table_has_rows")
    def test_stale_view_identified(self, mock_rows, mock_clients, mock_tables, mock_views, mock_location, mock_s3):
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_tables.return_value = ["my_view"]  # SHOW TABLES includes views
        mock_views.return_value = ["my_view"]
        mock_location.return_value = "s3://bucket/x/"
        mock_s3.return_value = True
        mock_rows.return_value = False  # view returns no rows

        summary = aws_athena_cleanup(database="db", workgroup="wg", region="us-west-2")
        assert "my_view" in summary["stale_views"]
        assert summary["stale_tables"] == []
