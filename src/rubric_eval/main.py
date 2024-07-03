"""
How to run:

1. Check out this repo locally
2. pip install -e .
3. python scripts/convert_dataset_to_rubriceval_input_format.py  # This will generate an example instructions.json
4. rubric_eval get_rubrics --input_path=instructions.json --output_path=instructions_with_rubrics.json
5. rubric_eval get_completions --model_configs="['gpt-4o-2024-05-13']" --input_path=instructions_with_rubrics.json --output_path_prefix=output_
6. rubric_eval --input_path_prefix=output_ --output_path_prefix=eval_output_
"""


import sys
import os
import ast
from pathlib import Path
import pandas as pd
from typing import List
from alpaca_eval import utils as ae_utils

import fire

from .helper import get_detailed_rubrics, get_model_completions, get_evaluations
from .annotators import RubricBrainstormer

CUR_DIR = Path(__file__).parent

__all__ = ["get_rubrics", "get_outputs", "eval_outputs"]


def check_df_fields(df, required_fields, optional_fields=set()):
    allowed_fields = required_fields | optional_fields
    
    actual_fields = set(df.columns)
    
    # Check if all required fields are present
    if not required_fields.issubset(actual_fields):
        missing = required_fields - actual_fields
        raise RuntimeError(f"Missing required fields: {missing}")
    
    # Check if there are any fields not in the allowed set
    extra_fields = actual_fields - allowed_fields
    if extra_fields:
        raise RuntimeError(f"Extra fields found: {extra_fields}")
    

def cleanup_df_rubrics(df_rubrics):
    # Drop rows where the "scoring_scales" column is not a dict (because sometimes rubric generator will refuse to generate rubric)
    mask = df_rubrics["scoring_scales"].apply(lambda x: isinstance(x, dict))
    return df_rubrics[mask]


def list_files_with_prefix(path_prefix, suffix):
    matching_files = []
    
    # Expand path_prefix to full absolute path
    full_path_prefix = os.path.abspath(path_prefix)
    
    # Get the directory part of the full_path_prefix
    prefix_dir = os.path.abspath(os.path.dirname(full_path_prefix))
    
    # Walk through the directory
    for root, _, files in os.walk(prefix_dir):
        for file in files:
            full_path = os.path.join(root, file)
            if (
                full_path == os.path.join(prefix_dir, file)
                and full_path.startswith(full_path_prefix)
                and full_path.endswith(suffix)
            ):
                matching_files.append(full_path)
    
    return matching_files


def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # Handle the case where the string is not a valid literal
        print(f"Error: '{s}' is not a valid list representation")
        return None



def get_rubrics(
    input_path: str = "instructions.json",
    output_path: str = "instructions_with_rubrics.json",
):
    """
    input_path: str, path to input file (JSON)
    The JSON file should contain a list of instructions, each instruction is a dictionary containing the following fields:
    - prompt
    - category (optional)
    - additional_information (optional, free-form text)

    TODO: should we allow more fields in the input JSON file that are just ignored, so that people don't need to manually remove those fields in their dataset JSON file before calling this function?
    TODO: should we support other serialization formats like pickle, Parquet, HDF5, etc.?
    TODO: should we support another Python function calling this function, thus needing to keep both input and output in memory (for perf reason) instead of being serialized to disk?
    TODO: `n_to_print` should be controlled by LOG_LEVEL. Also probably shouldn't print markdown stuff in CLI.
    TODO: should we allow customizing which model to use for rubric_generator (right now it's hardcoded to "gpt4_CoT_v0")?
    """
    print(f"Calling get_rubrics! input_path: {input_path}")
    assert input_path.endswith(".json"), "only JSON format is supported"
    df = pd.read_json(input_path)
    check_df_fields(
        df, 
        required_fields={"prompt"},
        optional_fields={"category", "additional_information"},
    )
    # Fill in missing fields
    for col in ["category", "additional_information"]:
        if col not in df.columns:
            df[col] = ""
    rubric_generator_name = "gpt4_CoT_v0"

    rubric_brainstormer = RubricBrainstormer(annotators_config=rubric_generator_name)
    criteria = rubric_brainstormer(df)
    df_criteria = rubric_brainstormer.make_df_rubrics(criteria)

    # TODO: this should probably be included in the post-processing of the rubric_brainstormer
    # copy "prompt" column to "final_prompt" column
    df_criteria["final_prompt"] = df_criteria["prompt"]
    
    df_rubrics = get_detailed_rubrics(df_criteria, n_to_print=3, is_store_missing_annotations=True, annotators_config=rubric_generator_name)
    df_rubrics = cleanup_df_rubrics(df_rubrics)
    df_rubrics.to_json(output_path, orient='records', indent=4)


def get_completions(
    model_configs: List[str] = [],
    input_path: str = "instructions_with_rubrics.json",
    output_path_prefix: str = f"output_"
):
    """
    For each model, we will save the completion to {output_path_prefix}{model_config}.json file.
    """
    print(f"Calling get_outputs! model_configs: {model_configs}")
    # model_configs = string_to_list(model_configs)
    assert input_path.endswith(".json"), "only JSON format is supported"
    df_rubrics = pd.read_json(input_path)
    check_df_fields(
        df_rubrics,
        # TODO: can we dedup "prompt" and "final_prompt"?
        required_fields={"prompt", "final_prompt", "category", "additional_information", "raw_completion", "scoring_scales", "criteria", "detailed_analytic_rubric"},
        # TODO: do we require any of these fields?
        optional_fields={'annotator', 'reference_answer', 'critique', 'time_per_example', 'clear_goals', 'price_per_example'},
    )
    # TODO: parallelize this?
    for model_config in model_configs:
        completions = get_model_completions(df_rubrics[["final_prompt", "category"]], model_config, n_to_print=3)
        df_completions = ae_utils.convert_to_dataframe(completions)
        df_rubrics = df_rubrics.reset_index(drop=True)
        df_completions = pd.concat([df_rubrics, df_completions], axis=1)
        df_completions = df_completions.loc[:, ~df_completions.columns.duplicated()]
        df_completions.to_json(f"{output_path_prefix}{model_config}.json", orient='records', indent=4)


def run_evals(
    input_path_prefix: str = "output_",
    output_path_prefix: str = "eval_output_",
):
    """
    For each model, we will:
    - Load the completions from {input_path_prefix}{model_name}.json file
    - Save the detailed evaluation results to {output_path_prefix}{model_name}.json file
    - Save the model card to {output_path_prefix}{model_name}_model_card.json file

    TODO: should we allow customizing evaluator_name (right now it's hardcoded to "gpt4_CoT_v0")?
    """
    
    print(f"Calling run_evals!")
    for input_path in list_files_with_prefix(input_path_prefix, suffix=".json"):
        print(f"Processing {input_path}...")
        model_name = input_path.replace(os.path.abspath(input_path_prefix), "").replace(".json", "")
        eval_result = {}
        df_completions = pd.read_json(input_path)
        check_df_fields(
            df_completions,
            # TODO: can we dedup "prompt" and "final_prompt"?
            required_fields={"prompt", "final_prompt", "category", "additional_information", "raw_completion", "scoring_scales", "criteria", "detailed_analytic_rubric"},
            # TODO: do we require any of these fields?
            optional_fields={'annotator', 'reference_answer', 'critique', 'time_per_example', 'clear_goals', 'price_per_example', 'output'},
        )
        evaluator_name = "gpt4_CoT_v0"
        df_evaluations = get_evaluations(df_completions, annotators_config=evaluator_name)
        eval_result["model_name"] = model_name
        eval_result["evaluator_name"] = evaluator_name
        eval_result["num_evaluations"] = len(df_evaluations)
        eval_result["mean_of_avg_score"] = df_evaluations["avg_score"].mean()
        eval_result["std_of_avg_score"] = df_evaluations["avg_score"].std()
        df_evaluations.to_json(f"{output_path_prefix}{model_name}.json", orient='records', indent=4)
        pd.DataFrame.from_dict(eval_result, orient='index').T.to_json(f"{output_path_prefix}{model_name}_model_card.json", orient='records', indent=4)


ALL_FUNCTIONS = {
    "get_rubrics": get_rubrics,
    "get_completions": get_completions,
    "run_evals": run_evals,
}


def main():
    is_fn_name = len(sys.argv) > 1 and "--" not in sys.argv[1]
    is_help = any(a == "--help" for a in sys.argv)

    if is_fn_name or is_help:
        fire.Fire(ALL_FUNCTIONS)
    else:
        # default behavior if no function is specified
        fire.Fire(run_evals)


if __name__ == "__main__":
    fire.Fire(ALL_FUNCTIONS)