import datetime
import importlib
import json
from concurrent.futures import Future
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from pyathena.pandas.result_set import AthenaPandasResultSet
from pyathena.sqlalchemy.base import AthenaDialect

from buildstock_query.schema.utilities import MappedColumn, QueryCompiler

KWH2MBTU = 0.003412141633127942
MBTU2KWH = 293.0710701722222
PANDAS_PARSERS = importlib.import_module("pandas._libs.parsers")


class CachedFutureDf(Future):
    def __init__(self, df: pd.DataFrame, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.df = df.copy()
        self.set_result(self.df)

    def running(self) -> Literal[False]:
        return False

    def done(self) -> Literal[True]:
        return True

    def cancelled(self) -> Literal[False]:
        return False

    def result(self, timeout=None) -> pd.DataFrame:
        return super().result(timeout=timeout)

    def as_df(self) -> pd.DataFrame:
        return self.df

    def as_pandas(self) -> pd.DataFrame:
        return self.df


class AthenaFutureDf:
    def __init__(self, db_future: Future) -> None:
        self.future = db_future

    def cancel(self) -> bool:
        return self.future.cancel()

    def running(self) -> bool:
        return self.future.running()

    def done(self) -> bool:
        return self.future.done()

    def cancelled(self) -> bool:
        return self.future.cancelled()

    def result(self, timeout=None) -> AthenaPandasResultSet:
        return self.future.result(timeout=timeout)

    def as_pandas(self) -> pd.DataFrame:
        df = self.future.as_df()  # type: ignore # mypy doesn't know about AthenaPandasResultSet
        return df


class COLOR:
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    END = '\033[0m'


def print_r(text):  # print in Red
    print(f"{COLOR.RED}{text}{COLOR.END}")


def print_y(text):  # print in Yellow
    print(f"{COLOR.YELLOW}{text}{COLOR.END}")


def print_g(text):  # print in Green
    print(f"{COLOR.GREEN}{text}{COLOR.END}")


class UnSupportedTypeException(Exception):
    pass


class CustomCompiler(AthenaDialect().statement_compiler):  # type: ignore

    @staticmethod
    def render_literal(obj):
        if isinstance(obj, (int, float)):
            return str(obj)
        elif isinstance(obj, str):
            escaped = obj.replace("'", "''")
            return f"'{escaped}'"
        elif isinstance(obj, (datetime.datetime)):
            escaped = str(obj).replace("'", "''")
            return f"timestamp '{escaped}'"
        elif isinstance(obj, list):
            return CustomCompiler.get_array_string(obj)
        elif isinstance(obj, tuple):
            return f"({', '.join([CustomCompiler.render_literal(v) for v in obj])})"
        elif isinstance(obj, MappedColumn):
            if obj.bsq is None:
                raise ValueError("MappedColumn must be associated with a BuildStockQuery instance.")
            compiler = cast(QueryCompiler, obj.bsq)
            keys = list(obj.mapping_dict.keys())
            values = list(obj.mapping_dict.values())
            if isinstance(obj.key, tuple):
                indexing_str = f"({', '.join(tuple(compiler._compile(source) for source in obj.key))})"
            else:
                indexing_str = compiler._compile(obj.key)

            return f"MAP({CustomCompiler.render_literal(keys)}, " +\
                   f"{CustomCompiler.render_literal(values)})[{indexing_str}]"
        else:
            raise UnSupportedTypeException(f"Unsupported type {type(obj)} for literal {obj}")

    @staticmethod
    def get_array_string(array):
        # rewrite to break into multiple arrays joined by CONCAT if the number of elements is > 254
        if len(array) > 254:
            array_list = ["ARRAY[" + ', '.join([CustomCompiler.render_literal(v) for v in array[i:i+254]]) + "]"
                          for i in range(0, len(array), 254)]
            return "CONCAT(" + ', '.join(array_list) + ")"
        else:
            return f"ARRAY[{', '.join([CustomCompiler.render_literal(v) for v in array])}]"

    def render_literal_value(self, obj, type_):
        if isinstance(obj, (datetime.datetime, list, tuple, MappedColumn)):
            return CustomCompiler.render_literal(obj)

        return super().render_literal_value(obj, type_)


class DataExistsException(Exception):
    def __init__(self, message, existing_data=None):
        super().__init__(message)
        self.existing_data = existing_data


def read_csv(csv_file_path, **kwargs) -> pd.DataFrame:
    default_na_values = cast(set[str], vars(PANDAS_PARSERS)["STR_NA_VALUES"])
    df = pd.read_csv(csv_file_path, na_values=list(default_na_values - {"None"}), keep_default_na=False, **kwargs)
    return df


def load_script_defaults(defaults_name):
    """
    Load the default input for script from cache
    """
    cache_folder = Path(".bsq_cache")
    cache_folder.mkdir(exist_ok=True)
    defaults_cache = cache_folder / f"{defaults_name}_defaults.json"
    defaults = {}
    if defaults_cache.exists():
        with open(defaults_cache) as f:
            defaults = json.load(f)
    return defaults


def save_script_defaults(defaults_name, defaults):
    """
    Save the current input for script to cache as the default for next run
    """
    cache_folder = Path(".bsq_cache")
    cache_folder.mkdir(exist_ok=True)
    defaults_cache = cache_folder / f"{defaults_name}_defaults.json"
    with open(defaults_cache, "w") as f:
        json.dump(defaults, f)
