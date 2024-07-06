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

from .helper import get_detailed_rubrics, get_model_completions, get_evaluations
from .annotators import RubricBrainstormer

CUR_DIR = Path(__file__).parent

__all__ = ["get_rubrics", "get_completions", "evaluate"]

logger = logging.getLogger(__name__)


def check_df_fields(df, required_fields):
    actual_fields = set(df.columns)
    
    # Check if all required fields are present
    if not required_fields.issubset(actual_fields):
        missing = required_fields - actual_fields
        raise RuntimeError(f"Missing required fields: {missing}")
    

def postprocess_df_rubrics(df_rubrics):
    orig_num_rows = len(df_rubrics)
    # Drop rows where the "scoring_scales" column is not a dict (because sometimes rubric generator will refuse to generate rubric)
    mask = df_rubrics["scoring_scales"].apply(lambda x: isinstance(x, dict))
    df_rubrics = df_rubrics[mask]
    logger.warn(f"Dropped {orig_num_rows - len(df_rubrics)} rows where 'scoring_scales' is not a dict")
    return df_rubrics


def get_rubrics(
    input_df: Union[pd.DataFrame, None] = None,
    rubric_generator = "gpt4_CoT_v0",  # TODO: is "gpt4_CoT_v0" the best default generator?
    *,
    input_path: AnyPath = "instructions.json",
    output_path: Union[AnyPath, None] = None,
) -> Union[pd.DataFrame, None]:
    """
    input_path: str, path to input file (JSON)
    The JSON file should contain a list of instructions, each instruction is a dictionary containing the following fields:
    - prompt
    - additional_information (optional, free-form text)
    """
    assert input_path.endswith(".json"), "only JSON format is supported"
    if input_df is not None:
        df = input_df
    else:
        df = ae_utils.load_or_convert_to_dataframe(input_path)
    check_df_fields(
        df, 
        required_fields={"prompt"},
    )
    # Fill in missing fields
    for col in ["additional_information"]:
        if col not in df.columns:
            df[col] = ""

    rubric_brainstormer = RubricBrainstormer(annotators_config=rubric_generator)
    criteria = rubric_brainstormer(df)
    df_criteria = rubric_brainstormer.make_df_rubrics(criteria)
    
    df_rubrics = get_detailed_rubrics(df_criteria, is_store_missing_annotations=True, annotators_config=rubric_generator)
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
):
    """
    For each model, we will save the completion to {output_path} JSON file.
    """
    assert input_path.endswith(".json"), "only JSON format is supported"
    if input_df is not None:
        df_rubrics = input_df
    else:
        df_rubrics = ae_utils.load_or_convert_to_dataframe(input_path)
    check_df_fields(
        df_rubrics,
        required_fields={"prompt", "additional_information", "raw_completion", "scoring_scales", "criteria", "detailed_analytic_rubric"},
    )
    completions = get_model_completions(df_rubrics, model_config)
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
):
    """
    For each model, we will:
    - Load the completions from {input_path} JSON file
    - Save the detailed evaluation results to {output_path} JSON file
    - Save the model card to model_card.json file in the same directory as {output_path}
    """
    
    eval_result = {}
    if input_df is not None:
        df_completions = input_df
    else:
        df_completions = ae_utils.load_or_convert_to_dataframe(input_path)
    check_df_fields(
        df_completions,
        required_fields={"prompt", "additional_information", "raw_completion", "scoring_scales", "criteria", "detailed_analytic_rubric"},
    )
    df_evaluations = get_evaluations(df_completions, annotators_config=evaluator)
    eval_result["model_name"] = model_config
    eval_result["evaluator"] = evaluator
    eval_result["num_evaluations"] = len(df_evaluations)
    eval_result["mean_of_avg_score"] = df_evaluations["avg_score"].mean()
    eval_result["std_of_avg_score"] = df_evaluations["avg_score"].std()
    df_eval_result = pd.DataFrame.from_dict(eval_result, orient='index').T
    logger.info("\n" + df_eval_result.to_string(formatters={'mean_of_avg_score': '{:.2f}'.format, 'std_of_avg_score': '{:.2f}'.format}))

    if input_df is not None:
        return df_evaluations
    else:
        if output_path is None:
            output_path = Path(input_path).parent / "evaluations.json"
        df_evaluations.to_json(output_path, orient='records', indent=4)
        model_card_path = Path(output_path).parent /  "model_card.json"
        df_eval_result.to_json(model_card_path, orient='records', indent=4)
        logger.info(f"Evaluations file is written to: {output_path}")
        logger.info(f"Model card file is written to: {model_card_path}")


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
    fire.Fire(ALL_FUNCTIONS)