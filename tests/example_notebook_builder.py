"""Generate runnable Jupyter notebooks from snapshot test entries.

Each (flavor.json, schema) pair becomes one notebook at
`tests/example_notebooks/<flavor>_<schema>.ipynb` containing:
  1. Imports
  2. BuildStockQuery construction (matched to the conftest fixture)
  3. One cell per entry that calls the entry's method with the entry's args
     and prints the head of the returned DataFrame

Notebook outputs are deterministic: cell IDs are content-derived hashes
(stable across regenerations), execution_count is None, and result
previews are truncated to head() so a fresh regen doesn't churn diffs
unless the underlying data actually changed.

Trigger: regeneration fires inside `evaluate_entries` when:
  - The notebook file is missing, OR
  - --update-snapshot / --overwrite-snapshot ran (data was just verified
    fresh, so the embedded preview should reflect it).
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

# Import lazily-needed pieces from test_utility at function scope to avoid a
# circular import (test_utility imports SNAPSHOTS_ROOT which lives there).


NOTEBOOKS_ROOT_NAME = "example_notebooks"

# Methods whose cells we COMMENT OUT during regen instead of letting
# nbclient execute them. These fan out to multiple internal Athena
# queries that the snapshot cache only partially covers (it stores the
# OUTER call's hash, not the inner per-upgrade probes). Determined by
# reading each method's body — only methods that issue more than one
# Athena query (loops, batch_query, multiple internal `_get_*` helpers
# that themselves fire queries) appear here.
#
# Cross-reference for adding new entries: in the method's source, look
# for `submit_batch_query`, multiple `self.execute(...)` calls, or
# loops over `available_upgrades`. Single-execute methods are safe.
_NO_EXECUTE_METHODS = frozenset({
    "report.get_success_report",       # _get_bs/_get_up/_get_change_report
                                       #   each fan out per-upgrade
    "report.check_ts_bs_integrity",    # one count-distinct per upgrade
    "agg.get_building_average_kws_at", # 3.2 TB landmine (see CLAUDE.md)
})


# Per-schema constructor knobs. Mirrors `tests/conftest.py:bsq_<schema>_oedi`
# so the notebook builds the same BuildStockQuery the test fixture did.
_SCHEMA_CONSTRUCTOR: dict[str, dict[str, Any]] = {
    "resstock_oedi": {
        "table_name": "resstock_2024_amy2018_release_2",
        "buildstock_type": "resstock",
        "db_schema": "resstock_oedi_vu",
    },
    "comstock_oedi": {
        "table_name": "comstock_amy2018_r2_2025",
        "buildstock_type": "comstock",
        "db_schema": "comstock_oedi_state_and_county",
    },
    "comstock_oedi_agg": {
        "table_name": "comstock_amy2018_r2_2025",
        "buildstock_type": "comstock",
        "db_schema": "comstock_oedi_agg_state_and_county",
    },
}


def _cell_id(content: str) -> str:
    """Stable cell ID derived from cell content. Notebooks re-render with the
    same ID for the same content, so git diffs only flip on real changes."""
    return hashlib.md5(content.encode()).hexdigest()[:8]


def _markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id(text),
        "metadata": {},
        "source": text.splitlines(keepends=True) or [""],
    }


def _code_cell(source: str, outputs: list[dict] | None = None) -> dict:
    """A code cell with deterministic ID and no execution_count.

    Outputs default to []; pass a populated list to embed a result preview.
    """
    return {
        "cell_type": "code",
        "id": _cell_id(source),
        "metadata": {},
        "execution_count": None,
        "outputs": outputs or [],
        "source": source.splitlines(keepends=True) or [""],
    }


def _execute_notebook_in_place(path: Path) -> None:
    """Run all cells in `path` via nbclient and write the result back.

    This populates outputs with whatever Jupyter would naturally produce
    (DataFrame HTML tables, log streams, etc.). Running the notebook in
    place means the saved file IS the runnable artifact AND the source of
    truth for what the demo currently outputs.

    Cells that legitimately can't execute (the `_NO_EXECUTE_METHODS` set,
    `_has_unroundtrippable_placeholder`) are rendered comment-only by the
    builder, so nbclient never sees them. Anything that does execute must
    succeed — a `CellExecutionError` propagates up so the caller (and
    pytest) can surface it.
    """
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel

    nb = nbformat.read(path, as_version=4)
    preferred_kernel = nb.get("metadata", {}).get("kernelspec", {}).get("name", "python3")
    kernel_mgr = KernelSpecManager()
    available_kernels = kernel_mgr.find_kernel_specs()
    kernel_name = preferred_kernel
    if kernel_name not in available_kernels:
        warnings.warn(
            f"Skipping notebook execution for {path.name}: kernel '{kernel_name}' is unavailable in this environment.",
            RuntimeWarning,
        )
        nbformat.write(nb, path)
        return
    # Run from the notebook's directory so its `_dh[0]`-based cache path
    # resolves to the sibling `tests/query_snapshots/` tree.
    client = NotebookClient(
        nb, timeout=300, kernel_name=kernel_name,
        resources={"metadata": {"path": str(path.parent)}},
    )
    try:
        client.execute()
    except NoSuchKernel:
        warnings.warn(
            f"Skipping notebook execution for {path.name}: kernel '{kernel_name}' is unavailable in this environment.",
            RuntimeWarning,
        )
    finally:
        # Persist outputs even on failure so the user can inspect the
        # crash trace in the notebook itself.
        nbformat.write(nb, path)


def _format_arg_value(v: Any) -> str:
    """Render an arg value as a Python literal string. SA-built objects
    (Label, MappedColumn) get a placeholder string — those aren't JSON-safe
    and the user has to construct them inline anyway. The `_applied_filter`,
    `_calc_column`, and `_mapped_column` sentinels render as inline
    construction calls so the notebook produces runnable Python."""
    if isinstance(v, list):
        return "[" + ", ".join(_format_arg_value(item) for item in v) + "]"
    if isinstance(v, tuple):
        if len(v) == 1:
            return "(" + _format_arg_value(v[0]) + ",)"
        return "(" + ", ".join(_format_arg_value(item) for item in v) + ")"
    if isinstance(v, dict):
        if "_applied_filter" in v:
            spec = v["_applied_filter"] or {}
            kwargs: list[str] = []
            if spec.get("any_of") is not None:
                kwargs.append(f"any_of={_format_arg_value(spec['any_of'])}")
            if spec.get("all_of") is not None:
                kwargs.append(f"all_of={_format_arg_value(spec['all_of'])}")
            return f"bsq.get_applied_buildings_filter({', '.join(kwargs)})"
        if "_calc_column" in v:
            spec = v["_calc_column"] or {}
            args = [
                _format_arg_value(spec["name"]),
                _format_arg_value(spec["expr"]),
                f"table={_format_arg_value(spec.get('table', 'baseline'))}",
            ]
            return f"bsq.get_calculated_column({', '.join(args)})"
        if "_mapped_column" in v:
            spec = v["_mapped_column"] or {}
            kwargs = [
                "bsq=bsq",
                f"name={_format_arg_value(spec['name'])}",
                f"mapping_dict={_format_arg_value(spec['mapping_dict'])}",
                f"key=bsq._get_column({_format_arg_value(spec['key_column'])})",
            ]
            return f"MappedColumn({', '.join(kwargs)})"
        return repr(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return repr(v)
    cls = type(v).__name__
    return f"<{cls} ...>"


def _has_unroundtrippable_placeholder(call_src: str) -> bool:
    """Detect args rendered as object reprs that aren't valid Python. Two
    sources:
      - `_format_arg_value`'s fallback: literal `<ClassName ...>` token
        (e.g. `<MappedColumn ...>`).
      - `repr(list_with_SA)`: when an arg is a `list` (e.g.
        `enduses=[label_obj]`), `_format_arg_value` returns `repr(v)` which
        per-element calls `repr(SA_obj)`, producing strings like
        `<sqlalchemy.sql.elements.Label at 0x...; name>` — also not valid
        Python. Such cells can't run as written; the user must construct
        the SA object inline."""
    # Match three repr shapes that indicate an unrunnable arg:
    #   - SA Label:   `<sqlalchemy.sql.elements.Label at 0x...; name>`
    #   - object():   `<buildstock_query.main.BuildStockQuery object at 0x...>`
    #   - fallback:   `<MappedColumn ...>` from _format_arg_value
    # The common shape is "<dotted.Name ... at 0x..." or "<ClassName ...>".
    import re
    if re.search(r"\bat\s+0x[0-9A-Fa-f]+", call_src):
        return True
    return bool(re.search(r"<[A-Z][A-Za-z_]+\s\.\.\.>", call_src))


def _render_call(method_path: str, args: Mapping[str, Any]) -> str:
    """Render `bsq.<method>(arg=val, ...)` as a one-line call. For long arg
    lists, split across lines for readability."""
    arg_strs = [f"{k}={_format_arg_value(v)}" for k, v in args.items()]
    inline = f"bsq.{method_path}(" + ", ".join(arg_strs) + ")"
    if len(inline) <= 100:
        return inline
    indented = ",\n    ".join(arg_strs)
    return f"bsq.{method_path}(\n    {indented},\n)"


def _build_notebook(
    *,
    schema: str,
    flavor: str,
    entries: Iterable,  # list[SnapshotEntry]
) -> dict:
    """Build the notebook dict (Jupyter v4 format)."""
    knobs = _SCHEMA_CONSTRUCTOR[schema]
    cells: list[dict] = []

    cells.append(_markdown_cell(
        f"# Example queries: `{flavor}` ({schema})\n"
        f"\n"
        f"Auto-generated from `tests/query_snapshots/{flavor}.json`. Each cell\n"
        f"runs one entry from the snapshot suite. Regenerate by running the\n"
        f"matching test with `--update-snapshot` or `--overwrite-snapshot`.\n"
    ))

    cells.append(_code_cell(
        "from pathlib import Path\n"
        "from buildstock_query import BuildStockQuery\n"
        "from buildstock_query.schema.utilities import MappedColumn\n"
        "import pandas as pd\n"
    ))

    cells.append(_markdown_cell(
        "## Construct the BuildStockQuery object\n"
        "\n"
        "`cache_folder` points at the snapshot test cache directory so this\n"
        "notebook reuses parquets that the test suite has already downloaded\n"
        "from Athena. Queries that are already cached return immediately;\n"
        "anything new still hits Athena.\n"
    ))

    cells.append(_code_cell(
        f'# This notebook lives in `tests/example_notebooks/`; the snapshot test\n'
        f'# cache is its sibling `tests/query_snapshots/{schema}_cache/`. Resolve\n'
        f'# the path relative to the notebook directory (`_dh[0]` is set by\n'
        f'# IPython at kernel startup; falls back to CWD outside Jupyter).\n'
        f'_NB_DIR = Path(_dh[0] if "_dh" in globals() else ".").resolve()\n'
        f'_CACHE = (_NB_DIR / "../query_snapshots/{schema}_cache").resolve()\n'
        f'bsq = BuildStockQuery(\n'
        f'    "rescore",\n'
        f'    "buildstock_sdr",\n'
        f'    "{knobs["table_name"]}",\n'
        f'    buildstock_type="{knobs["buildstock_type"]}",\n'
        f'    db_schema="{knobs["db_schema"]}",\n'
        f'    skip_reports=True,\n'
        f'    cache_folder=str(_CACHE),\n'
        f')\n'
    ))

    from tests.test_utility import expand_rate_map_flat

    for entry in entries:
        # The entry has been placeholder-resolved already by load_entries(),
        # so entry.args is a list[dict] of fully-concrete kwargs.
        if not entry.args:
            continue
        first_variant = expand_rate_map_flat(dict(entry.args[0]))  # copy — we mutate
        method_name = first_variant.pop("_method", None) or "query"

        # Header for this entry.
        header = f"## `{entry.name}`\n"
        if entry.description:
            header += f"\n{entry.description}\n"
        cells.append(_markdown_cell(header))

        call_src = _render_call(method_name, first_variant)
        unsafe = method_name in _NO_EXECUTE_METHODS
        not_roundtrippable = _has_unroundtrippable_placeholder(call_src)
        if unsafe or not_roundtrippable:
            # Comment-out cases:
            #   - `unsafe`: this method fans out to internal queries that
            #     aren't covered by the snapshot cache and would fire
            #     fresh Athena scans during nbclient execution.
            #   - `not_roundtrippable`: the call contains a `<Label ...>`
            #     or `<MappedColumn ...>` placeholder because the arg
            #     was an SA expression that couldn't be rendered as a
            #     Python literal. The user must construct it inline.
            commented = "\n".join("# " + line for line in call_src.splitlines())
            if unsafe:
                why = (
                    f"# NOTE: cell intentionally not executed — `{method_name}` issues\n"
                    f"# additional Athena queries beyond the snapshot test cache.\n"
                    f"# Uncomment to run live (incurs scan cost):\n"
                )
            else:
                why = (
                    f"# NOTE: cell intentionally not executed — call contains a\n"
                    f"# placeholder (`<Label ...>` or `<MappedColumn ...>`) that\n"
                    f"# can't be auto-generated. To run, replace the placeholder\n"
                    f"# with a live `bsq.get_calculated_column(...)` or\n"
                    f"# `MappedColumn(...)` construction:\n"
                )
            full_src = f"{why}{commented}\n"
        else:
            # Cell ends with `result.head()` so Jupyter emits a DataFrame
            # preview as the cell's `execute_result` when the notebook is run.
            full_src = f"result = {call_src}\nresult.head() if hasattr(result, 'head') else result\n"
        cells.append(_code_cell(full_src))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook_for_flavor(
    *,
    schema: str,
    flavor: str,
    entries: list,  # list[SnapshotEntry]
    snapshots_root: Path,
    execute: bool = True,
) -> Path:
    """Render and write `tests/example_notebooks/<flavor>_<schema>.ipynb`.

    When `execute=True` (the default), the notebook is run through nbclient
    after writing so its outputs reflect actual query results. Set False to
    write source-only notebooks (faster regen, but cells appear "never run").
    """
    if schema not in _SCHEMA_CONSTRUCTOR:
        raise ValueError(f"Unknown schema for notebook generation: {schema}")
    if not entries:
        # Don't write an empty notebook.
        return snapshots_root.parent / NOTEBOOKS_ROOT_NAME / f"{flavor}_{schema}.ipynb"

    nb = _build_notebook(schema=schema, flavor=flavor, entries=entries)

    out_dir = snapshots_root.parent / NOTEBOOKS_ROOT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{flavor}_{schema}.ipynb"
    out_path.write_text(json.dumps(nb, indent=1, sort_keys=False) + "\n")

    if execute:
        _execute_notebook_in_place(out_path)
    return out_path


def notebook_path_for_flavor(*, schema: str, flavor: str, snapshots_root: Path) -> Path:
    return snapshots_root.parent / NOTEBOOKS_ROOT_NAME / f"{flavor}_{schema}.ipynb"
