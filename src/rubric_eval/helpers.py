import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import datasets
import pandas as pd
from alpaca_eval import constants as ae_const
from alpaca_eval import utils as ae_utils
from alpaca_eval.types import AnyPath

CONFIGS_DIR = Path(__file__).parent / "configs"
MAIN_DIR = Path(__file__).parents[2]


def mean(x):
    return sum(x) / len(x)


def percentage_scale_df_(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Scale a certain column in the df so that it's normalized in percentage (i.e. divide by sum)."""
    for col in columns:
        df[col] = df[col].apply(lambda x: x / df[col].sum()) * 100
    return df


def get_output_path(
    input_path: AnyPath,
    output_path: Optional[AnyPath],
    to_rm: Sequence[str] = [],
    prfx: str = "",
    sffx: str = "",
    is_rm_double_underscore: bool = True,
    extension: str = ".json",
) -> Path:
    """Get the output path by potentially using the input path with the new suffix."""
    input_path = Path(input_path)
    if output_path is None:
        output_stem = input_path.stem
        for rm in to_rm:
            output_stem = output_stem.replace(rm, "")
        output_stem = prfx + output_stem + sffx
        if is_rm_double_underscore:
            output_stem = output_stem.replace("__", "_")
        output_path = input_path.with_name(output_stem + extension)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return output_path


def literal_eval_if_string(x: Any) -> Any:
    """Literal eval if the input is a string."""
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except Exception as e:
        return ""


def expand_json_column(
    df: pd.DataFrame,
    column_name: str,
    is_keep_other_columns: bool = True,
    is_add_prefix: bool = False,
) -> pd.DataFrame:
    """Expands a column of JSON strings into separate columns.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        column_name (str): The name of the column to expand.
        is_keep_other_columns (bool): Whether to keep the other columns in the DataFrame.
        is_add_prefix (bool): whether to add `column_name` as a prefix to the new columns from the
            json. Note that this will not be added if the column already starts with the `column_name`.

    Example:
        >>> df = pd.DataFrame({"a": [1, 2], "b": ['{"c": 3, "d": 1}', '{"c": 4, "d": 2}']})
        >>> expand_json_column(df, "b")
                a	c	d
            0	1	3	1
            1	2	4	2
    """
    mask = df[column_name] == ""
    if mask.any():
        logging.warning(f"{mask.sum()} examples have empty {column_name} and will be dropped.")
        df = df[~mask]

    df[column_name] = df[column_name].apply(literal_eval_if_string)

    # convert the annotated string to a dictionary
    mask = df[column_name] == ""
    if mask.any():
        logging.warning(
            f"{mask.sum()} examples have not been able to be parsed from {column_name} and will be dropped."
        )
        df = df[~mask]

    # Splits up the columns from the annotation key into separate columns
    columns_to_add = pd.json_normalize(df[column_name], max_level=0)
    if is_add_prefix:
        columns_to_add.columns = [
            column_name + "_" + col if not col.startswith(column_name) else col for col in columns_to_add.columns
        ]
    return pd.concat(
        ([df.drop([column_name], axis=1)] if is_keep_other_columns else []) + [columns_to_add],
        axis=1,
    )


def add_suffix_to_path(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix + ".json")


def process_input_df_(df, required_fields=None, optional_fields=None):
    """
    Preprocess the DataFrame inplace.

    This function checks if the input DataFrame contains the required fields and fills in optional_fields with defaults
    inplace.

    Args:
        df (pd.DataFrame): Input DataFrame.
        required_fields (Set[str]): Set of required field names.
        optional_fields (Dict[str, Any]): Dictionary of optional field names and their default values.

    Raises:
        RuntimeError: If any required fields is missing in the DataFrame.
    """
    check_df_fields(
        df,
        required_fields=required_fields or set(),
    )

    # Fill in missing fields
    optional_fields = optional_fields or dict()
    for col, default in optional_fields.items():
        if col not in df.columns:
            df[col] = default
    return df


def check_df_fields(df, required_fields):
    """
    Check if a DataFrame contains all the required fields.

    This function compares the columns of the input DataFrame with the set of required fields
    and raises a RuntimeError if any required field is missing.

    Args:
        df (pd.DataFrame): Input DataFrame to check.
        required_fields (Set[str]): Set of required field names.

    Raises:
        RuntimeError: If any required field is missing in the DataFrame.
    """
    actual_fields = set(df.columns)

    # Check if all required fields are present
    if not required_fields.issubset(actual_fields):
        missing = required_fields - actual_fields
        raise RuntimeError(f"Missing required fields: {missing}")


def dict_reverser(d):
    return {v: k for k, v in d.items()}
