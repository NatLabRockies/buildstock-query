"""
Search for a table across all AWS Glue/Athena databases.

Scans all databases in the Glue catalog and reports which ones contain a table
matching the given name (exact or substring match).

Usage:
    aws_athena_table_search --table my_table_name
    aws_athena_table_search --table my_table --substring
    aws_athena_table_search --table my_table --region us-east-1
"""

import argparse
import re
from typing import Optional

from botocore.exceptions import ClientError

from .aws_utils import get_glue_client, list_databases


def search_table_in_database(
    glue_client,
    database: str,
    table_name: str,
    substring: bool = False,
    regex: bool = False,
) -> list[dict]:
    """
    Search for a table in a specific database.

    Returns a list of matching table info dicts with keys:
    'database', 'table', 's3_location', 'create_time'.
    """
    matches = []
    paginator = glue_client.get_paginator("get_tables")

    try:
        for page in paginator.paginate(DatabaseName=database):
            for table in page["TableList"]:
                name = table["Name"]
                matched = False

                if regex:
                    matched = bool(re.search(table_name, name))
                elif substring:
                    matched = table_name.lower() in name.lower()
                else:
                    matched = name.lower() == table_name.lower()

                if matched:
                    s3_location = (
                        table.get("StorageDescriptor", {}).get("Location", "")
                    )
                    create_time = table.get("CreateTime")
                    matches.append(
                        {
                            "database": database,
                            "table": name,
                            "s3_location": s3_location or "(none)",
                            "create_time": str(create_time) if create_time else "(unknown)",
                        }
                    )
    except ClientError as e:
        if e.response["Error"]["Code"] == "AccessDeniedException":
            print(f"  WARNING: Access denied for database '{database}', skipping.")
        else:
            raise

    return matches


def aws_athena_table_search(
    table_name: str,
    region: str = "us-west-2",
    substring: bool = False,
    regex: bool = False,
    database_filter: Optional[str] = None,
) -> list[dict]:
    """
    Search for a table across all Glue databases.

    Parameters
    ----------
    table_name : str
        Table name to search for (exact, substring, or regex depending on flags).
    region : str
        AWS region.
    substring : bool
        If True, match any table whose name contains `table_name` as a substring.
    regex : bool
        If True, treat `table_name` as a regex pattern.
    database_filter : str, optional
        If provided, only search databases whose name contains this substring.

    Returns
    -------
    list[dict]
        List of matches, each with keys: 'database', 'table', 's3_location', 'create_time'.
    """
    glue_client = get_glue_client(region)

    databases = list_databases(glue_client)
    if database_filter:
        databases = [db for db in databases if database_filter.lower() in db.lower()]

    print(f"\n{'='*60}")
    print("AWS ATHENA TABLE SEARCH")
    print(f"{'-'*60}")
    print(f"  Search term:  {table_name}")
    print(f"  Mode:         {'regex' if regex else 'substring' if substring else 'exact'}")
    print(f"  Region:       {region}")
    print(f"  Databases:    {len(databases)}" + (f" (filtered by '{database_filter}')" if database_filter else ""))
    print(f"{'='*60}\n")

    all_matches = []
    for i, db in enumerate(databases, 1):
        print(f"  [{i}/{len(databases)}] Searching '{db}'...", end="")
        matches = search_table_in_database(glue_client, db, table_name, substring, regex)
        if matches:
            print(f" FOUND {len(matches)} match(es)")
            all_matches.extend(matches)
        else:
            print(" -")

    # --- Results ---
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'-'*60}")
    if all_matches:
        for m in all_matches:
            print(f"  {m['database']}.{m['table']}")
            print(f"    S3:      {m['s3_location']}")
            print(f"    Created: {m['create_time']}")
    else:
        print("  No matches found.")
    print(f"\n  Total matches: {len(all_matches)}\n")
    print(f"{'='*60}")

    return all_matches


def main():
    parser = argparse.ArgumentParser(
        description="Search for a table across all Athena/Glue databases."
    )
    parser.add_argument("-t", "--table", required=True, help="Table name to search for.")
    parser.add_argument("-r", "--region", default="us-west-2", help="AWS region (default: us-west-2).")
    parser.add_argument("-s", "--substring", action="store_true",
                        help="Match tables containing the search term as a substring.")
    parser.add_argument("-E", "--regex", action="store_true",
                        help="Treat the search term as a regex pattern.")
    parser.add_argument("-f", "--database-filter", default=None,
                        help="Only search databases whose name contains this substring.")
    args = parser.parse_args()

    aws_athena_table_search(
        table_name=args.table,
        region=args.region,
        substring=args.substring,
        regex=args.regex,
        database_filter=args.database_filter,
    )


if __name__ == "__main__":
    main()
