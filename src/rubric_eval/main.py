"""
How to run:

1. git clone https://github.com/stella-z3g/RubricEval -b public_release
2. cd RubricEval/ && pip install -e .
3. python scripts/convert_dataset_to_rubriceval_input_format.py  # This will generate an example instructions.json
4. rubric_eval generate_rubrics --input_path=instructions.json
5. rubric_eval generate_outputs --model_configs=gpt-4o-2024-05-13 --input_path=instructions_with_rubrics.json
6. rubric_eval --model_configs=gpt-4o-2024-05-13 --input_path=outputs.json
"""


import logging
import sys
from pathlib import Path
from typing import Optional, Union

import fire
import pandas as pd
from alpaca_eval import utils as ae_utils
from alpaca_eval.types import AnyPath

from rubric_eval import Evaluator, Outputer, Rubricator, RubricBrainstormer
from rubric_eval.helper import check_df_fields, get_output_path, process_input_df_

CUR_DIR = Path(__file__).parent

__all__ = [
    "brainstorm_rubrics",
    "generate_rubrics",
    "generate_outputs",
    "evaluate",
    "brainstorm_and_generate_rubrics",
    "generate_outputs_and_evaluate",
    # "generate_report",
]

logger = logging.getLogger(__name__)


def brainstorm_rubrics(  # from path
    input_path: AnyPath,
    output_path: Optional[AnyPath] = None,
    brainstormer_configs: AnyPath = "gpt-4o-2024-08-06_CoT_v0",
    is_rm_prev_columns: bool = True,
    **brainstormer_kwargs,
):
    """Generate brainstormed rubrics for given instructions.

    Args:
        input_path (AnyPath): Path to the file containing instructions. If the file is JSON, then it should be a list
            of dictionaries where each dictionary represents an instruction and contains a key "instruction" and
            optionally "useful_info_to_eval_instruction" to improve the rubric brainstorming.
            Any other key will be ignored but still be present in the output. More generally the input can be read by
            pd.read_json, pd.read_csv, or pd.read_table.
        output_path (Optional[AnyPath], optional): Path to save the output JSON file containing instructions
            with brainstormed rubrics. The output JSON adds a key "brainstormed_rubric" to each instruction dictionary.
            If None, the output will be saved in the same directory as the input file with "_with_brainstorm.json"
            appended to the filename.
        brainstormer_configs (AnyPath, optional): Path to the configuration directory for the rubric brainstormer. The
            directory should contains a `configs.yaml` file. The path can be absolute or relative to
            `rubric_eval/configs/rubric_brainstormer_configs`.
        is_rm_prev_columns (bool, optional): Whether to remove unecessary columns from previous steps (e.g., learning
            objectives and useful_info_to_eval_instruction).
        **brainstormer_kwargs: Additional keyword arguments to pass to the rubric brainstormer.
    """
    df = ae_utils.load_or_convert_to_dataframe(input_path)
    df_with_brainstorm = brainstorm_rubrics_from_df(
        df, brainstormer_configs, is_rm_prev_columns=is_rm_prev_columns, **brainstormer_kwargs
    )
    output_path = get_output_path(input_path, output_path, sffx="_with_brainstorm")
    df_with_brainstorm.to_json(output_path, orient="records", indent=4)
    logger.info(f"Instructions with brainstormed rubrics are written to: {output_path}")


def brainstorm_rubrics_from_df(
    df_input: pd.DataFrame,
    brainstormer_configs: str = "gpt-4o-2024-08-06_CoT_v0",
    is_rm_prev_columns: bool = True,
    **brainstormer_kwargs,
) -> pd.DataFrame:
    """Same as brainstorm_rubrics but takes a DataFrame as input and output."""
    process_input_df_(
        df_input,
        required_fields={"instruction"},
        optional_fields={"useful_info_to_eval_instruction": ""},
    )
    rubric_brainstormer = RubricBrainstormer(annotators_config=brainstormer_configs, **brainstormer_kwargs)
    brainstormed_rubric = rubric_brainstormer(df_input)
    # converts rubric from string to list of dict, and renormalize weight as percentage
    df_brainstormed = rubric_brainstormer.make_df_rubrics(
        brainstormed_rubric, is_renormalize_weight=True, is_extract_criteria_col=False
    )
    # df_brainstormed = df_brainstormed.rename(columns={"annotator": "brainstormer"})
    if is_rm_prev_columns:
        df_brainstormed = df_brainstormed.drop(
            columns=["learning_objectives", "useful_info_to_eval_instruction"], errors="ignore"
        )
    return df_brainstormed


def generate_rubrics(
    input_path: AnyPath,
    output_path: Optional[AnyPath] = None,
    rubricator_configs: AnyPath = "gpt-4o-2024-08-06_CoT_v0",
    is_rm_prev_columns: bool = True,
    **rubricator_kwargs,
):
    """Generate detailed rubrics for given instructions.

    Args:
        input_path (AnyPath): Path to the file containing instructions. If the file is JSON, then it should be a list
            of dictionaries where each dictionary represents an instruction and contains a key "instruction" and
            "brainstormed_rubric". Any other key will be ignored but still be present in the output. More generally the
            input can be read by pd.read_json, pd.read_csv, or pd.read_table.
        output_path (Optional[AnyPath], optional): Path to save the output JSON file containing instructions
            with generated rubrics. If None, the output will be saved in the same directory as the input file with
            "_with_rubrics.json" appended to the filename (and without "_with_brainstorm" if present).
        rubricator_configs (AnyPath, optional): Path to the configuration directory for the rubric generator. The
            directory should contains a `configs.yaml` file. The path can be absolute or relative to
            `rubric_eval/configs/rubricator_configs`.
        is_rm_prev_columns (bool, optional): Whether to remove unecessary columns from previous steps (e.g., brainstormed
            rubrics and learning_objectives).
        **rubricator_kwargs: Additional keyword arguments to pass to the rubricator
    """
    df = ae_utils.load_or_convert_to_dataframe(input_path)
    df_with_rubrics = generate_rubrics_from_df(
        df, rubricator_configs, is_rm_prev_columns=is_rm_prev_columns, **rubricator_kwargs
    )
    output_path = get_output_path(input_path, output_path, sffx="_with_rubrics", to_rm=["_with_brainstorm"])
    df_with_rubrics.to_json(output_path, orient="records", indent=4)
    logger.info(f"Instructions with generated rubrics are written to: {output_path}")


def generate_rubrics_from_df(
    df_input: pd.DataFrame,
    rubricator_configs: str = "gpt-4o-2024-08-06_CoT_v0",
    is_rm_prev_columns: bool = True,
    **rubricator_kwargs,
) -> pd.DataFrame:
    """Same as generate_rubrics but takes a DataFrame as input and output."""
    process_input_df_(df_input, required_fields={"instruction", "brainstormed_rubric"})
    rubric_generator = Rubricator(annotators_config=rubricator_configs, **rubricator_kwargs)
    list_rubrics = rubric_generator(df_input)
    df_rubrics = rubric_generator.make_df_rubrics(
        list_rubrics, is_extract_criteria_col=True, is_renormalize_weight=True
    )
    # df_rubrics = df_rubrics.rename(columns={"annotator": "rubricator"})
    if is_rm_prev_columns:
        df_rubrics = df_rubrics.drop(
            columns=["brainstormed_rubric", "learning_objectives", "useful_info_to_eval_instruction"], errors="ignore"
        )
    return df_rubrics


def brainstorm_and_generate_rubrics(
    input_path: AnyPath,
    output_path: Optional[AnyPath] = None,
    brainstormer_configs: AnyPath = "gpt-4o-2024-08-06_CoT_v0",
    rubricator_configs: AnyPath = "gpt-4o-2024-08-06_CoT_v0",
    **brainstormer_and_rubricator_kwargs,
):
    """Applies brainstorming and rubric generation to instructions."""
    df = ae_utils.load_or_convert_to_dataframe(input_path)
    df_with_brainstorm = brainstorm_rubrics_from_df(df, brainstormer_configs, **brainstormer_and_rubricator_kwargs)
    df_with_rubrics = generate_rubrics_from_df(
        df_with_brainstorm, rubricator_configs, **brainstormer_and_rubricator_kwargs
    )
    output_path = get_output_path(input_path, output_path, sffx="_with_rubrics")
    df_with_rubrics.to_json(output_path, orient="records", indent=4)
    logger.info(f"Instructions with generated rubrics are written to: {output_path}")


def generate_outputs(
    input_path: AnyPath,
    output_path: Optional[AnyPath] = None,
    model_configs: AnyPath = "gpt-4o-mini-2024-07-18",
    is_rm_prev_columns: bool = True,
    **outputer_kwargs,
):
    """Generate model outputs for given instructions.

    Args:
        input_path (AnyPath): Path to the file containing instructions. If the file is JSON, then it should be a list
            of dictionaries where each dictionary represents an instruction and contains a key "instruction" and
            "rubric". Any other key will be ignored but still be present in the output. More generally
            the input can be read by pd.read_json, pd.read_csv, or pd.read_table.
        output_path (Optional[AnyPath], optional): Path to save the output JSON file containing instructions
            with generated outputs. If None, the output will be saved in the same directory as the input file with
            "_with_outputs.json" appended to the filename.
        model_configs (AnyPath, optional): Path to the configuration directory for the model. The directory should contains
            a `configs.yaml` file. The path can be absolute or relative to `rubric_eval/configs/model_configs` or
            `alpaca_eval/model_configs`. In particular, you can evaluate any model from the alpaca_eval library.
        is_rm_prev_columns (bool, optional): Whether to remove unecessary columns from previous steps (e.g., brainstormed
            rubrics and learning_objectives).
        **outputer_kwargs: Additional keyword arguments to pass to the outputer.
    """
    df = ae_utils.load_or_convert_to_dataframe(input_path)
    breakpoint()
    df_outputs = generate_outputs_from_df(df, model_configs, is_rm_prev_columns=is_rm_prev_columns, **outputer_kwargs)
    output_path = get_output_path(input_path, output_path, prfx="outputs_", to_rm=["instructions_", "instructions"])
    df_outputs.to_json(output_path, orient="records", indent=4)
    logger.info(f"Instructions with generated rubrics are written to: {output_path}")


def generate_outputs_from_df(
    df_input: pd.DataFrame,
    model_configs: str = "gpt-4o-mini-2024-07-18",
    is_rm_prev_columns: bool = True,
    **outputer_kwargs,
) -> pd.DataFrame:
    """Same as generate_rubrics but takes a DataFrame as input and output."""
    process_input_df_(df_input, required_fields={"instruction"})
    outputer = Outputer(annotators_config=model_configs, **outputer_kwargs)
    outputs = outputer(df_input)
    df_outputs = ae_utils.convert_to_dataframe(outputs)
    # df_outputs = df_outputs.rename(columns={"annotator": "model"})
    if is_rm_prev_columns:
        df_outputs = df_outputs.drop(
            columns=["brainstormed_rubric", "learning_objectives", "useful_info_to_eval_instruction"], errors="ignore"
        )
    return df_outputs


def evaluate(
    input_path: AnyPath,
    output_path: Optional[AnyPath] = None,
    evaluator_configs: str = "gpt-4o-2024-08-06_CoT_v0",
    is_rm_prev_columns: bool = True,
    **evaluator_kwargs,
):
    """Evaluate model outputs using generated rubrics.

    Args:
        input_path (AnyPath): Path to the file containing model outputs and rubrics. If the file is JSON, then it
            should be a list of dictionaries where each dictionary represents a completion and contains keys
            "instruction", "output", "rubric". Any other key will be ignored but still be present in the output.
            More generally the input can be read by pd.read_json, pd.read_csv, or pd.read_table.
        output_path (Optional[AnyPath], optional): Path to save the output JSON file containing evaluation results.
            If None, the output will be saved in the same directory as the input file with "_evaluations.json" appended
            to the filename.
        evaluator_configs (str, optional): Configuration for the evaluator. Defaults to "gpt-4o-2024-08-06_CoT_v0".
        is_rm_prev_columns (bool, optional): Whether to remove unecessary columns from previous steps (e.g., brainstormed
            rubrics and learning_objectives).
        **evaluator_kwargs: Additional keyword arguments to pass to the evaluator.
    """
    df = ae_utils.load_or_convert_to_dataframe(input_path)
    df_evaluations = evaluate_from_df(df, evaluator_configs, is_rm_prev_columns=is_rm_prev_columns, **evaluator_kwargs)
    output_path = get_output_path(input_path, output_path, sffx="_with_evaluations", to_rm=["_with_rubrics"])
    df_evaluations.to_json(output_path, orient="records", indent=4)
    logger.info(f"Evaluation results are written to: {output_path}")


def evaluate_from_df(
    df_input: pd.DataFrame,
    evaluator_configs: str = "gpt-4o-2024-08-06_CoT_v0",
    is_rm_prev_columns: bool = True,
    **evaluator_kwargs,
) -> pd.DataFrame:
    """Same as evaluate but takes a DataFrame as input and output."""
    process_input_df_(df_input, required_fields={"instruction", "output", "rubric", "criteria"})
    evaluator = Evaluator(annotators_config=evaluator_configs, **evaluator_kwargs)
    evaluations = evaluator(df_input)
    df_eval = evaluator.make_df_rubric_grading(evaluations)
    # df_eval = df_eval.rename(columns={"annotator": "evaluator"})
    if is_rm_prev_columns:
        df_eval = df_eval.drop(
            columns=["brainstormed_rubric", "learning_objectives", "useful_info_to_eval_instruction", "criteria"],
            errors="ignore",
        )
    return df_eval


def generate_outputs_and_evaluate(
    input_path: AnyPath,
    output_path: Optional[AnyPath] = None,
    model_configs: str = "gpt-4o-mini-2024-07-18",
    evaluator_configs: str = "gpt-4o-2024-08-06_CoT_v0",
    **completor_and_evaluator_kwargs,
):
    """Generate model outputs and evaluate them using generated rubrics."""
    df = ae_utils.load_or_convert_to_dataframe(input_path)
    df_with_outputs = generate_outputs_from_df(df, model_configs, **completor_and_evaluator_kwargs)
    df_evaluations = evaluate_from_df(df_with_outputs, evaluator_configs, **completor_and_evaluator_kwargs)
    output_path = get_output_path(
        input_path,
        output_path,
        prfx="outputs_",
        sffx="_with_evaluations",
        to_rm=["instructions_", "instructions", "_with_rubrics", "with_rubrics"],
    )
    df_evaluations.to_json(output_path, orient="records", indent=4)
    logger.info(f"Evaluation results are written to: {output_path}")


# make report

# def evaluate(
#     model_configs: str,
#     df_input: Union[pd.DataFrame, None] = None,
#     evaluator: str = "gpt-4o-2024-08-06_CoT_v0",
#     *,
#     input_path: AnyPath = "outputs.json",
#     output_path: Union[AnyPath, None] = None,
#     cache_dir: Union[AnyPath, None] = "auto",
# ):
#     """
#     Evaluate model outputs using generated rubrics.
#
#     This function takes a DataFrame or JSON file containing model outputs and rubrics,
#     evaluates the outputs using the specified evaluator, and produces evaluation results.
#
#     Args:
#         model_configs (str): Configuration of the model being evaluated.
#         df_input (Union[pd.DataFrame, None], optional): Input DataFrame containing outputs
#             and rubrics. If provided, the function will use this DataFrame instead of loading
#             from a file. Defaults to None.
#         evaluator (str, optional): Configuration for the evaluator. Defaults to "gpt-4o-2024-08-06_CoT_v0".
#         input_path (AnyPath, optional): Path to the input JSON file containing outputs and
#             rubrics. Defaults to "outputs.json".
#         output_path (Union[AnyPath, None], optional): Path to save the output JSON files
#             containing evaluation results and model card. If None, the output will not be saved
#             to files. Defaults to None.
#         cache_dir (Union[AnyPath, None], optional): Path to cache the annotations to.
#             If None, will not save the annotations. If the path already exists it will
#             load annotations from there. Defaults to "auto".
#
#     Returns:
#         Union[pd.DataFrame, None]: DataFrame containing evaluation results. If output_path is
#         provided, the function will save the output to files and return None.
#
#     >>> # Test with DataFrame input and output
#     >>> import pandas as pd
#     >>> from pathlib import Path
#     >>> from os.path import dirname, abspath
#     >>> outputs_path = Path(__file__).resolve().parent.parent.parent / 'tests' / 'test_data' / 'outputs.json'
#     >>> df_outputs = pd.read_json(outputs_path)
#     >>> df_evaluations, df_model_card = evaluate("gpt-4o-2024-05-13", df_outputs)
#     >>> 'criteria_scores' in df_evaluations.columns
#     True
#     >>> 'mean_of_avg_score' in df_model_card.columns
#     True
#
#     >>> # Test with file input and output
#     >>> import tempfile, os, json
#     >>> from pathlib import Path
#     >>> from os.path import dirname, abspath
#     >>> with tempfile.TemporaryDirectory() as tmpdir:
#     ...     outputs_path = Path(__file__).resolve().parent.parent.parent / 'tests' / 'test_data' / 'outputs.json'
#     ...     evaluations_path = Path(tmpdir) / 'evaluations.json'
#     ...     evaluate("gpt-4o-2024-05-13", input_path=outputs_path, output_path=evaluations_path)
#     ...     df_evaluations = pd.read_json(evaluations_path)
#     ...     'criteria_scores' in df_evaluations.columns
#     True
#     ...     model_card_path = Path(tmpdir) / 'model_card.json'
#     ...     df_model_card = pd.read_json(model_card_path)
#     ...     'mean_of_avg_score' in df_model_card.columns
#     True
#     """
#
#     eval_result = {}
#     if df_input is not None:
#         df_outputs = df_input
#     else:
#         df_outputs = ae_utils.load_or_convert_to_dataframe(input_path)
#     check_df_fields(
#         df_outputs,
#         required_fields={"instruction", "useful_info_to_eval_instruction", "output", "scoring_scales", "criteria", "detailed_analytic_rubric"},
#     )
#     df_evaluations = get_evaluations(df_outputs, annotators_config=evaluator, cache_dir=cache_dir)
#     eval_result["model_name"] = model_configs
#     eval_result["evaluator"] = evaluator
#     eval_result["num_evaluations"] = len(df_evaluations)
#     eval_result["mean_of_avg_score"] = df_evaluations["avg_score"].mean()
#     eval_result["std_of_avg_score"] = df_evaluations["avg_score"].std()
#     df_eval_result = pd.DataFrame.from_dict(eval_result, orient='index').T
#     logger.info("\n" + df_eval_result.to_string(formatters={'mean_of_avg_score': '{:.2f}'.format, 'std_of_avg_score': '{:.2f}'.format}))
#
#     if df_input is not None:
#         return df_evaluations, df_eval_result
#     else:
#         if output_path is None:
#             output_path = Path(input_path).parent / "evaluations.json"
#         df_evaluations.to_json(output_path, orient='records', indent=4)
#         model_card_path = Path(output_path).parent /  "model_card.json"
#         df_eval_result.to_json(model_card_path, orient='records', indent=4)
#         logger.info(f"Evaluations file is written to: {output_path}")
#         logger.info(f"Model card file is written to: {model_card_path}")


def get_instructions(
    n_max_examples: int,
    category: str | None = None,
    instruction_set: str = "auto",
    with_additional_info: bool = False,
    random_seed: int = 123,
    output_path: Optional[AnyPath] = None,
    cache_dir: str = "auto",
) -> Optional[pd.DataFrame]:
    assert instruction_set in ["auto", "alpaca_eval_2", "wildbench-v1"]
    if instruction_set == "alpaca_eval_2":
        # TODO(stella): this branch doesn't seem to work: `BuilderConfig 'v1-legacy' not found. Available: ['alpaca_eval', 'alpaca_eval_gpt4_baseline', 'alpaca_eval_all_outputs', 'alpaca_farm_human_annotations', 'alpaca_farm_human_crossannotations', 'alpaca_eval_annotations_alpaca_eval_gpt4', 'alpaca_eval_annotations_claude']`
        instructions = ae_const.ALPACAEVAL_REFERENCE_OUTPUTS_2()
        df_instructions = instructions.to_pandas().sample(n_max_examples, random_state=random_seed)
        df_instructions["instruction"] = df_instructions["instruction"]
        df_instructions["category"] = df_instructions["dataset"]
    elif instruction_set == "wildbench-v1":
        ds = datasets.load_dataset("allenai/WildBench", "v1-legacy")["test"]
        ds = ds.to_pandas()
        ds = ds[
            ds.apply(lambda x: len(x["conversation_input"]) == 1, axis=1)
        ]  # maintain single-turn instructions for now
        df_instructions = pd.DataFrame()
        df_instructions["instruction"] = ds["conversation_input"].apply(lambda x: x[0]["content"])
        df_instructions["category"] = ds["category"]
        if category:
            n_max_examples = min(
                n_max_examples,
                len(df_instructions[df_instructions["category"] == category]),
            )
            df_instructions = df_instructions[df_instructions["category"] == category].sample(
                n_max_examples, random_state=random_seed
            )
        else:
            df_instructions = df_instructions.sample(n_max_examples, random_state=random_seed)
        if with_additional_info:
            # merge the intent and checklist into the useful_info_to_eval_instruction
            def _get_useful_info_to_eval_instruction(x):
                info = ""
                info += f"User intent: {x['intent']}\n"
                checklist = "- " + "\n- ".join(x["checklist"].tolist())
                info += f"Reference checklist:\n{checklist}\n"

                return info

            df_instructions["useful_info_to_eval_instruction"] = ds.apply(_get_useful_info_to_eval_instruction, axis=1)
        else:
            df_instructions["useful_info_to_eval_instruction"] = "N/A"
    if output_path is not None:
        df_instructions.to_json(output_path, orient="records", indent=4)
    else:
        return df_instructions


def get_evaluations(outputs, cache_dir: str = "auto", **kwargs):
    caching_path = cache_dir
    if cache_dir is not None and cache_dir != "auto":
        caching_path = Path(cache_dir) / "evaluator_configs.json"
    evaluator = Evaluator(caching_path=caching_path, **kwargs)
    scores = evaluator(outputs)
    df_scores = ae_utils.convert_to_dataframe(scores).query("not output.isin(['', ' '])")
    df_scores = df_scores.dropna(subset=["criteria_scores"])

    def calc_mean_with_exception_handling(x):
        x_value_set = set(x.values())
        if "N/A" in x_value_set or "n/a" in x_value_set:
            return None
        else:
            try:
                return pd.Series(x).mean()
            except:
                print(f"x: {x}")
                raise

    df_scores["avg_score"] = df_scores["criteria_scores"].apply(calc_mean_with_exception_handling)
    df_scores = df_scores[df_scores["avg_score"].notnull()]

    def get_criteria_annotation(s):
        ret = {}
        for k, v in s["criteria_scores"].items():
            if v == 0 or v is None:
                ret[k] = None
            else:
                ret[k] = dict_reverser(s["scoring_scales"])[v]
        return ret

    df_criteria_annotations = df_scores.apply(
        get_criteria_annotation,
        axis=1,
    )
    assert not df_criteria_annotations.empty
    df_scores["criteria_annotations"] = df_criteria_annotations
    return df_scores


ALL_FUNCTIONS = {
    "brainstorm_rubrics": brainstorm_rubrics,
    "generate_rubrics": generate_rubrics,
    "generate_outputs": generate_outputs,
    "evaluate": evaluate,
    "generate_outputs_and_evaluate": generate_outputs_and_evaluate,
    "brainstorm_and_generate_rubrics": brainstorm_and_generate_rubrics,
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
