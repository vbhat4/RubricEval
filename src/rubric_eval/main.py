"""
How to run:

1. git clone https://github.com/stella-z3g/RubricEval -b public_release
2. cd RubricEval/ && pip install -e .
3. python scripts/convert_dataset_to_rubriceval_input_format.py  # This will generate an example instructions.json
4. rubric_eval get_rubrics --input_path=instructions.json
5. rubric_eval get_completions --model_config=gpt-4o-2024-05-13 --input_path=instructions_with_rubrics.json
6. rubric_eval --model_config=gpt-4o-2024-05-13 --input_path=completions.json
"""


import sys
import os
import ast
from pathlib import Path
from alpaca_eval.types import AnyPath
import pandas as pd
from typing import List, Union
from alpaca_eval import utils as ae_utils
from pathlib import Path
import logging

import fire

from rubric_eval.helper import get_detailed_rubrics, get_model_completions, get_evaluations
from rubric_eval.annotators import RubricBrainstormer

CUR_DIR = Path(__file__).parent

__all__ = ["get_rubrics", "get_completions", "evaluate"]

logger = logging.getLogger(__name__)


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
    

def preprocess_df_instructions(df):
    """
    Preprocess the instructions DataFrame.

    This function checks if the input DataFrame contains the required fields and fills
    in missing fields with empty strings.

    Args:
        df (pd.DataFrame): Input DataFrame containing instructions.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with missing fields filled in.

    Raises:
        RuntimeError: If any required fields is missing in the DataFrame.
    """
    check_df_fields(
        df,
        required_fields={"prompt"},
    )
    # Fill in missing fields
    for col in ["additional_information"]:
        if col not in df.columns:
            df[col] = ""
    return df


def postprocess_df_rubrics(df_rubrics):
    """
    Postprocess the rubrics DataFrame by dropping invalid rows.

    Args:
        df_rubrics (pd.DataFrame): Input DataFrame containing rubrics.

    Returns:
        pd.DataFrame: Postprocessed DataFrame with invalid rows dropped.
    """
    orig_num_rows = len(df_rubrics)
    # Drop rows where the "scoring_scales" column is not a dict (because sometimes rubric generator will refuse to generate rubric)
    mask = df_rubrics["scoring_scales"].apply(lambda x: isinstance(x, dict))
    df_rubrics = df_rubrics[mask]
    num_dropped_rows = orig_num_rows - len(df_rubrics)
    if num_dropped_rows > 0:
        logger.warn(f"Dropped {num_dropped_rows} rows where 'scoring_scales' is not a dict")
    return df_rubrics


def get_rubrics(
    input_df: Union[pd.DataFrame, None] = None,
    rubric_generator = "gpt4_CoT_v0",  # TODO: is "gpt4_CoT_v0" the best default generator? or maybe use gpt4o?
    *,
    input_path: AnyPath = "instructions.json",
    output_path: Union[AnyPath, None] = None,
    cache_dir: Union[AnyPath, None] = "auto",
) -> Union[pd.DataFrame, None]:
    """
    Generate detailed rubrics for given instructions.

    This function takes a DataFrame or JSON file containing instructions and generates 
    detailed rubrics for evaluating completions based on those instructions.

    Args:
        input_df (Union[pd.DataFrame, None], optional): Input DataFrame containing instructions. 
            If provided, the function will use this DataFrame instead of loading from a file. 
            Defaults to None.
        rubric_generator (str, optional): Configuration for the rubric generator. 
            Defaults to "gpt4_CoT_v0".
        input_path (AnyPath, optional): Path to the input JSON file containing instructions. 
            Defaults to "instructions.json".
        output_path (Union[AnyPath, None], optional): Path to save the output JSON file 
            containing instructions with generated rubrics. If None, the output will not be 
            saved to a file. Defaults to None.
        cache_dir (Union[AnyPath, None], optional): Path to cache the annotations to.
            If None, will not save the annotations. If the path already exists it will
            load annotations from there. Defaults to "auto".

    Returns:
        Union[pd.DataFrame, None]: DataFrame containing instructions with generated rubrics. 
        If output_path is provided, the function will save the output to a file and return None.

    >>> # Test with DataFrame input and output
    >>> import pandas as pd
    >>> df_instructions = pd.DataFrame({'prompt': ['Write a short story about a cat.']})
    >>> df_rubrics = get_rubrics(df_instructions)
    >>> 'scoring_scales' in df_rubrics.columns
    True

    >>> # Test with file input and output
    >>> import tempfile, os, json
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     instructions_path = Path(tmpdir) / 'instructions.json'
    ...     with open(instructions_path, 'w') as f:
    ...         json.dump([{'prompt': 'Write a short story about a cat.'}], f)
    ...     instructions_with_rubrics_path = Path(tmpdir) / 'instructions_with_rubrics.json'
    ...     get_rubrics(input_path=instructions_path, output_path=instructions_with_rubrics_path)
    ...     output_df = pd.read_json(instructions_with_rubrics_path)
    ...     'scoring_scales' in output_df.columns
    True
    """
    assert str(input_path).endswith(".json"), "only JSON format is supported"
    if input_df is not None:
        df = input_df
    else:
        df = ae_utils.load_or_convert_to_dataframe(input_path)
    df = preprocess_df_instructions(df)
    caching_path = cache_dir
    if cache_dir is not None and cache_dir != "auto":
        caching_path = Path(cache_dir) / "rubric_brainstormer_configs.json"
    rubric_brainstormer = RubricBrainstormer(annotators_config=rubric_generator, caching_path=caching_path)
    criteria = rubric_brainstormer(df)
    df_criteria = rubric_brainstormer.make_df_rubrics(criteria)
    
    df_rubrics = get_detailed_rubrics(df_criteria, is_store_missing_annotations=True, annotators_config=rubric_generator, cache_dir=cache_dir)
    df_rubrics = postprocess_df_rubrics(df_rubrics)
    if input_df is not None:
        return df_rubrics
    else:
        if output_path is None:
            output_path = input_path.replace(".json", "_with_rubrics.json")
        df_rubrics.to_json(output_path, orient='records', indent=4)
        logger.info(f"Instructions with rubrics file is written to: {output_path}")


def get_completions(
    model_config: str,
    input_df: Union[pd.DataFrame, None] = None,
    *,
    input_path: AnyPath = "instructions_with_rubrics.json",
    output_path: Union[AnyPath, None] = None,
    cache_dir: Union[AnyPath, None] = "auto",
):
    """
    Generate model completions for given instructions and rubrics.

    This function takes a DataFrame or JSON file containing instructions with rubrics and
    generates model completions based on the provided model configuration.

    Args:
        model_config (str): Configuration for the model to generate completions.
        input_df (Union[pd.DataFrame, None], optional): Input DataFrame containing instructions
            with rubrics. If provided, the function will use this DataFrame instead of loading 
            from a file. Defaults to None.
        input_path (AnyPath, optional): Path to the input JSON file containing instructions
            with rubrics. Defaults to "instructions_with_rubrics.json".
        output_path (Union[AnyPath, None], optional): Path to save the output JSON file 
            containing generated completions. If None, the output will not be saved to a file.
            Defaults to None.
        cache_dir (Union[AnyPath, None], optional): Path to cache the annotations to.
            If None, will not save the annotations. If the path already exists it will
            load annotations from there. Defaults to "auto".

    Returns:
        Union[pd.DataFrame, None]: DataFrame containing generated completions. If output_path 
        is provided, the function will save the output to a file and return None.

    >>> # Test with DataFrame input and output
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> from os.path import dirname, abspath
    >>> instructions_with_rubrics_path = Path(__file__).resolve().parent.parent.parent / 'tests' / 'test_data' / 'instructions_with_rubrics.json'
    >>> df_rubrics = pd.read_json(instructions_with_rubrics_path)
    >>> df_completions = get_completions("gpt-4o-2024-05-13", df_rubrics)
    >>> 'output' in df_completions.columns
    True

    >>> # Test with file input and output
    >>> import tempfile, os, json
    >>> from pathlib import Path
    >>> from os.path import dirname, abspath
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     instructions_with_rubrics_path = Path(__file__).resolve().parent.parent.parent / 'tests' / 'test_data' / 'instructions_with_rubrics.json'
    ...     completions_path = Path(tmpdir) / 'completions.json'
    ...     get_completions("gpt-4o-2024-05-13", input_path=instructions_with_rubrics_path, output_path=completions_path)
    ...     df_completions = pd.read_json(completions_path)
    ...     'output' in df_completions.columns
    True
    """
    assert str(input_path).endswith(".json"), "only JSON format is supported"
    if input_df is not None:
        df_rubrics = input_df
    else:
        df_rubrics = ae_utils.load_or_convert_to_dataframe(input_path)
    check_df_fields(
        df_rubrics,
        required_fields={"prompt", "additional_information", "scoring_scales", "criteria", "detailed_analytic_rubric"},
    )
    completions = get_model_completions(df_rubrics, model_config, cache_dir=cache_dir)
    df_completions = ae_utils.convert_to_dataframe(completions)
    if input_df is not None:
        return df_completions
    else:
        if output_path is None:
            output_path = Path(input_path).parent / "completions.json"
        df_completions.to_json(output_path, orient='records', indent=4)
        logger.info(f"Completions file is written to: {output_path}")


def evaluate(
    model_config: str,
    input_df: Union[pd.DataFrame, None] = None,
    evaluator: str = "gpt4_CoT_v0",
    *,
    input_path: AnyPath = "completions.json",
    output_path: Union[AnyPath, None] = None,
    cache_dir: Union[AnyPath, None] = "auto",
):
    """
    Evaluate model completions using generated rubrics.

    This function takes a DataFrame or JSON file containing model completions and rubrics,
    evaluates the completions using the specified evaluator, and produces evaluation results.

    Args:
        model_config (str): Configuration of the model being evaluated.
        input_df (Union[pd.DataFrame, None], optional): Input DataFrame containing completions 
            and rubrics. If provided, the function will use this DataFrame instead of loading
            from a file. Defaults to None.
        evaluator (str, optional): Configuration for the evaluator. Defaults to "gpt4_CoT_v0".
        input_path (AnyPath, optional): Path to the input JSON file containing completions and
            rubrics. Defaults to "completions.json".
        output_path (Union[AnyPath, None], optional): Path to save the output JSON files 
            containing evaluation results and model card. If None, the output will not be saved 
            to files. Defaults to None.
        cache_dir (Union[AnyPath, None], optional): Path to cache the annotations to.
            If None, will not save the annotations. If the path already exists it will
            load annotations from there. Defaults to "auto".

    Returns:
        Union[pd.DataFrame, None]: DataFrame containing evaluation results. If output_path is
        provided, the function will save the output to files and return None.
    
    >>> # Test with DataFrame input and output
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> from os.path import dirname, abspath
    >>> completions_path = Path(__file__).resolve().parent.parent.parent / 'tests' / 'test_data' / 'completions.json'
    >>> df_completions = pd.read_json(completions_path)
    >>> df_evaluations, df_model_card = evaluate("gpt-4o-2024-05-13", df_completions)
    >>> 'criteria_scores' in df_evaluations.columns
    True
    >>> 'mean_of_avg_score' in df_model_card.columns
    True

    >>> # Test with file input and output
    >>> import tempfile, os, json
    >>> from pathlib import Path
    >>> from os.path import dirname, abspath
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     completions_path = Path(__file__).resolve().parent.parent.parent / 'tests' / 'test_data' / 'completions.json'
    ...     evaluations_path = Path(tmpdir) / 'evaluations.json'
    ...     evaluate("gpt-4o-2024-05-13", input_path=completions_path, output_path=evaluations_path)
    ...     df_evaluations = pd.read_json(evaluations_path)
    ...     'criteria_scores' in df_evaluations.columns
    True
    ...     model_card_path = Path(tmpdir) / 'model_card.json'
    ...     df_model_card = pd.read_json(model_card_path)
    ...     'mean_of_avg_score' in df_model_card.columns
    True
    """
    
    eval_result = {}
    if input_df is not None:
        df_completions = input_df
    else:
        df_completions = ae_utils.load_or_convert_to_dataframe(input_path)
    check_df_fields(
        df_completions,
        required_fields={"prompt", "additional_information", "output", "scoring_scales", "criteria", "detailed_analytic_rubric"},
    )
    df_evaluations = get_evaluations(df_completions, annotators_config=evaluator, cache_dir=cache_dir)
    eval_result["model_name"] = model_config
    eval_result["evaluator"] = evaluator
    eval_result["num_evaluations"] = len(df_evaluations)
    eval_result["mean_of_avg_score"] = df_evaluations["avg_score"].mean()
    eval_result["std_of_avg_score"] = df_evaluations["avg_score"].std()
    df_eval_result = pd.DataFrame.from_dict(eval_result, orient='index').T
    logger.info("\n" + df_eval_result.to_string(formatters={'mean_of_avg_score': '{:.2f}'.format, 'std_of_avg_score': '{:.2f}'.format}))

    if input_df is not None:
        return df_evaluations, df_eval_result
    else:
        if output_path is None:
            output_path = Path(input_path).parent / "evaluations.json"
        df_evaluations.to_json(output_path, orient='records', indent=4)
        model_card_path = Path(output_path).parent /  "model_card.json"
        df_eval_result.to_json(model_card_path, orient='records', indent=4)
        logger.info(f"Evaluations file is written to: {output_path}")
        logger.info(f"Model card file is written to: {model_card_path}")


def run_doctests():
    import doctest
    test_results = doctest.testmod(verbose=True)
    assert test_results.attempted > 0, "No doctests were run!"


ALL_FUNCTIONS = {
    "get_rubrics": get_rubrics,
    "get_completions": get_completions,
    "evaluate": evaluate,
}


def main():
    is_fn_name = len(sys.argv) > 1 and "--" not in sys.argv[1]
    is_help = any(a == "--help" for a in sys.argv)

    if is_fn_name or is_help:
        fire.Fire(ALL_FUNCTIONS)
    else:
        # default behavior if no function is specified
        fire.Fire(evaluate)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run_doctests":
        run_doctests()
    else:
        fire.Fire(ALL_FUNCTIONS)
