"""
How to run:

1. git clone https://github.com/stella-z3g/RubricEval -b public_release
2. cd RubricEval/ && pip install -e .
3. python scripts/convert_dataset_to_rubriceval_input_format.py  # This will generate an example instructions.json
4. rubric_eval generate_rubrics --input_path=instructions.json
5. rubric_eval generate_outputs --model_configs=gpt-4o-2024-05-13 --input_path=instructions_with_rubrics.json
6. rubric_eval --model_configs=gpt-4o-2024-05-13 --input_path=outputs.json
"""


import json
import logging
import sys
import re
from pathlib import Path
from typing import Any, Optional, Union

import fire
import numpy as np
import pandas as pd
from alpaca_eval import utils as ae_utils
from alpaca_eval.types import AnyPath

from rubric_eval import (
    Evaluator,
    Outputer,
    Rubricator,
    RubricBrainstormer,
    format_evaluation_report_md,
    make_evaluation_report_dict,
    summarize,
)

from .helpers import check_df_fields, get_output_path, process_input_df_

CUR_DIR = Path(__file__).parent

__all__ = [
    "brainstorm_rubrics",
    "generate_rubrics",
    "generate_outputs",
    "evaluate",
    "generate_report",
    "brainstorm_and_generate_rubrics",
    "evaluate_and_generate_report",
    "generate_outputs_and_evaluation_report",
]

logger = logging.getLogger(__name__)


def brainstorm_rubrics(  # from path
    input_path: AnyPath,
    output_path: Optional[AnyPath] = None,
    brainstormer_configs: AnyPath = "gpt-4o-2024-08-06_CoT_v0",
    is_rm_prev_columns: bool = True,
    **brainstormer_kwargs,
) -> Path:
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
    return output_path


def brainstorm_rubrics_from_df(
    df_input: pd.DataFrame,
    brainstormer_configs: str = "gpt-4o-2024-08-06_CoT_v0",
    is_rm_prev_columns: bool = True,
    max_instances: Optional[int] = None,
    **brainstormer_kwargs,
) -> pd.DataFrame:
    """Same as brainstorm_rubrics but takes a DataFrame as input and output."""
    if max_instances:
        n_inputs = len(df_input)
        df_input = df_input.sample(min(max_instances, len(df_input)), random_state=123)
        logging.info(f"We sampled {len(df_input)} from the {n_inputs} due to max_instances.")

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
    if is_rm_prev_columns:
        raw_completion_cols = [c for c in df_brainstormed.columns if c.endswith("raw_completion")]
        df_brainstormed = df_brainstormed.drop(columns=["learning_objectives"] + raw_completion_cols, errors="ignore")
    return df_brainstormed


def generate_rubrics(
    input_path: AnyPath,
    output_path: Optional[AnyPath] = None,
    rubricator_configs: AnyPath = "gpt-4o-2024-08-06_CoT_v0",
    is_rm_prev_columns: bool = True,
    **rubricator_kwargs,
) -> Path:
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
    return output_path


def generate_rubrics_from_df(
    df_input: pd.DataFrame,
    rubricator_configs: str = "gpt-4o-2024-08-06_CoT_v0",
    is_rm_prev_columns: bool = True,
    max_instances: Optional[int] = None,
    **rubricator_kwargs,
) -> pd.DataFrame:
    """Same as generate_rubrics but takes a DataFrame as input and output."""
    if max_instances:
        n_inputs = len(df_input)
        df_input = df_input.sample(min(max_instances, len(df_input)), random_state=123)
        logging.info(f"We sampled {len(df_input)} from the {n_inputs} due to max_instances.")

    process_input_df_(
        df_input,
        required_fields={"instruction", "brainstormed_rubric"},
        optional_fields={"brainstormed_response": ""},
    )
    rubricator = Rubricator(annotators_config=rubricator_configs, **rubricator_kwargs)
    list_rubrics = rubricator(df_input)

    df_rubrics = rubricator.make_df_rubrics(list_rubrics, is_extract_criteria_col=True, is_renormalize_weight=True)

    if is_rm_prev_columns:
        raw_completion_cols = [c for c in df_rubrics.columns if c.endswith("raw_completion")]
        df_rubrics = df_rubrics.drop(
            columns=[
                "brainstormed_rubric",
                "brainstormed_response",
                "learning_objectives",
                "useful_info_to_eval_instruction",
            ]
            + raw_completion_cols,
            errors="ignore",
        )
    return df_rubrics


def brainstorm_and_generate_rubrics(
    input_path: AnyPath,
    output_path: Optional[AnyPath] = None,
    brainstormer_configs: AnyPath = "gpt-4o-2024-08-06_CoT_v0",
    rubricator_configs: AnyPath = "gpt-4o-2024-08-06_CoT_v0",
    **brainstormer_and_rubricator_kwargs,
) -> Path:
    """Applies brainstorming and rubric generation to instructions."""
    df = ae_utils.load_or_convert_to_dataframe(input_path)
    df_with_brainstorm = brainstorm_rubrics_from_df(df, brainstormer_configs, **brainstormer_and_rubricator_kwargs)
    df_with_rubrics = generate_rubrics_from_df(
        df_with_brainstorm, rubricator_configs, **brainstormer_and_rubricator_kwargs
    )
    output_path = get_output_path(input_path, output_path, sffx="_with_rubrics")
    df_with_rubrics.to_json(output_path, orient="records", indent=4)
    logger.info(f"Instructions with generated rubrics are written to: {output_path}")
    return output_path


def generate_outputs(
    model_configs: Optional[AnyPath],
    input_path: AnyPath,
    output_path: Optional[AnyPath] = None,
    is_rm_prev_columns: bool = True,
    dataset_name: Optional[str] = None,
    **outputer_kwargs,
) -> Path:
    """Generate model outputs for given instructions.

    Args:
        input_path (AnyPath): Path to the file containing instructions. If the file is JSON, then it should be a list
            of dictionaries where each dictionary represents an instruction and contains a key "instruction" and
            "rubric". Any other key will be ignored but still be present in the output. More generally
            the input can be read by pd.read_json, pd.read_csv, or pd.read_table.
        output_path (Optional[AnyPath], optional): Path to save the output JSON file containing instructions
            with generated outputs. If None, the output will be saved in the same directory as the input file with
            "_with_outputs.json" appended to the filename.
        model_configs (AnyPath): Path to the configuration directory for the model. The directory should contains
            a `configs.yaml` file. The path can be absolute or relative to `rubric_eval/configs/model_configs` or
            `alpaca_eval/model_configs`. In particular, you can evaluate any model from the alpaca_eval library.
            If `None`, the function will skip generating outputs and raise an error if input_df doesn't contain an
            "output" column.
        is_rm_prev_columns (bool, optional): Whether to remove unnecessary columns from previous steps (e.g., brainstormed
            rubrics and learning_objectives).
        dataset_name: Optional[str]: Name of the benchmark. If provided, it will be added as a column to the output.
            If not provided, the benchmark name will be the input path.
        **outputer_kwargs: Additional keyword arguments to pass to the outputer.
    """
    df = ae_utils.load_or_convert_to_dataframe(input_path)
    dataset_name = _get_dataset_name(input_path, dataset_name)
    if model_configs is None:
        if "output" not in df.columns:
            raise ValueError("`model_configs=None` but the input input DataFrame doesn't contain an 'output' column.")
        return input_path

    df_outputs = generate_outputs_from_df(
        df, model_configs, is_rm_prev_columns=is_rm_prev_columns, dataset_name=dataset_name, **outputer_kwargs
    )
    output_path = get_output_path(input_path, output_path, prfx="outputs_", to_rm=["instructions_", "instructions"])
    df_outputs.to_json(output_path, orient="records", indent=4)
    logger.info(f"Instructions with generated rubrics are written to: {output_path}")
    return output_path


def _get_dataset_name(input_path: AnyPath, dataset_name: Optional[str]) -> Optional[str]:
    """Get the benchmark name from the input path or the provided name."""
    if dataset_name is None:
        dataset_name = input_path

    try:
        full_path = Path(dataset_name)
        dataset_name = f"{full_path.parent.name}/{full_path.name}"
    except Exception:
        dataset_name = None

    return dataset_name


def generate_outputs_from_df(
    df_input: pd.DataFrame,
    model_configs: str = "gpt-4o-mini-2024-07-18",
    is_rm_prev_columns: bool = True,
    dataset_name: Optional[Union[AnyPath, str]] = None,  # name or path
    max_instances: Optional[int] = None,
    **outputer_kwargs,
) -> pd.DataFrame:
    """Same as generate_rubrics but takes a DataFrame as input and output."""
    if max_instances:
        n_inputs = len(df_input)
        df_input = df_input.sample(min(max_instances, len(df_input)), random_state=123)
        logging.info(f"We sampled {len(df_input)} from the {n_inputs} due to max_instances.")

    process_input_df_(df_input, required_fields={"instruction"})
    outputer = Outputer(annotators_config=model_configs, **outputer_kwargs)
    outputs = outputer(df_input)
    df_outputs = ae_utils.convert_to_dataframe(outputs)
    if is_rm_prev_columns:
        raw_completion_cols = [c for c in df_outputs.columns if c.endswith("raw_completion")]
        df_outputs = df_outputs.drop(
            columns=[
                "brainstormed_rubric",
                "brainstormed_response",
                "learning_objectives",
                "useful_info_to_eval_instruction",
            ]
            + raw_completion_cols,
            errors="ignore",
        )
    if dataset_name:
        df_outputs["dataset_name"] = dataset_name
    # TODO: remove hard coding
    model = df_outputs["model"].iloc[0]
    if model == "DeepSeek-R1":
        df_outputs["output"] = df_outputs["output"].apply(lambda x: re.sub(r"^<think>.*?</think>\s*", "", x, flags = re.DOTALL))
    return df_outputs


def evaluate(
    input_path: AnyPath,
    output_path: Optional[AnyPath] = None,
    evaluator_configs: str = "gpt-4o-2024-08-06_CoT_v0",
    is_rm_prev_columns: bool = True,
    **evaluator_kwargs,
) -> Path:
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
    df_evaluations = evaluate_from_df(
        df,
        evaluator_configs,
        is_rm_prev_columns=is_rm_prev_columns,
        **evaluator_kwargs,
    )
    output_path = get_output_path(input_path, output_path, sffx="_with_evaluations", to_rm=["_with_rubrics"])
    df_evaluations.to_json(output_path, orient="records", indent=4)
    logger.info(f"Evaluation results are written to: {output_path}")
    return output_path


def evaluate_from_df(
    df_input: pd.DataFrame,
    evaluator_configs: str = "gpt-4o-2024-08-06_CoT_v0",
    is_rm_prev_columns: bool = True,
    max_instances: Optional[int] = None,
    **evaluator_kwargs,
) -> pd.DataFrame:
    """Same as evaluate but takes a DataFrame as input and output."""
    if max_instances:
        n_inputs = len(df_input)
        df_input = df_input.sample(min(max_instances, len(df_input)), random_state=123)
        logging.info(f"We sampled {len(df_input)} from the {n_inputs} due to max_instances.")

    process_input_df_(
        df_input,
        required_fields={"instruction", "output", "rubric", "criteria"},
        optional_fields={"excellent_response": ""},
    )

    evaluator = Evaluator(annotators_config=evaluator_configs, **evaluator_kwargs)
    evaluations = evaluator(df_input)
    df_eval = evaluator.make_df_rubric_grading(evaluations)
    if is_rm_prev_columns:
        raw_completion_cols = [c for c in df_eval.columns if c.endswith("raw_completion")]
        df_eval = df_eval.drop(
            columns=[
                "brainstormed_rubric",
                "brainstormed_response",
                "learning_objectives",
                "useful_info_to_eval_instruction",
                "criteria",
            ]
            + raw_completion_cols,
            errors="ignore",
        )
    return df_eval


def generate_report(
    input_path: AnyPath,
    output_path: Optional[AnyPath] = None,
    summarizer_configs: Optional[str] = "unstructured_gpt-4o-2024-08-06",
    **summarizer_kwargs,
):
    """Generate a report from the evaluations.

    Args:
        input_path (AnyPath): Path to the file containing evaluation results. If the file is JSON, then it should be a
            list of dictionaries where each dictionary represents an evaluation and contains keys "instruction",
            "output", "rubric", "criteria", "evaluation". Any other key will be ignored but still be present in the
            output. More generally the input can be read by pd.read_json, pd.read_csv, or pd.read_table.
        output_path (Optional[AnyPath], optional): Path to save the output report. If None, the output will be saved in
            the same directory as the input file with "_report.md" appended to the filename.
        summarizer_configs (Optional[str], optional): Configuration for the summarizer. If `None` then we skip the
            summarization step and the report will not have any qualitative analysis.
        **report_kwargs: Additional keyword arguments to pass to the report generator.
    """
    df = ae_utils.load_or_convert_to_dataframe(input_path)
    report_dict, report_str = generate_report_from_df(
        df,
        summarizer_configs,
        **summarizer_kwargs,
    )
    report_json_path = _save_reports(report_str, report_dict, input_path, output_path)
    return report_json_path


def _save_reports(report_str: str, report_json: dict, input_path: AnyPath, output_path: AnyPath) -> Path:
    path_kwargs = dict(prfx="report_evaluations_", to_rm=["_with_evaluations"])
    report_md_path = get_output_path(input_path, output_path, extension=".md", **path_kwargs)
    report_json_path = get_output_path(input_path, output_path, extension=".json", **path_kwargs)

    with open(report_md_path, "w") as f:
        f.write(report_str)

    with open(report_json_path, "w") as f:
        json.dump(report_json, f, indent=4)

    logger.info(f"Evaluation report is written to: {report_md_path} (formatted) and {report_json_path} (raw).")
    return report_json_path


def generate_report_from_df(
    df_input: pd.DataFrame,
    summarizer_configs: Optional[str] = "unstructured_gpt-4o-2024-08-06",
    max_instances: Optional[int] = None,
    **summarizer_kwargs,
) -> dict[str, Any]:
    """Same as generate_report but takes a DataFrame as input and output."""
    if max_instances:
        n_inputs = len(df_input)
        df_input = df_input.sample(min(max_instances, len(df_input)), random_state=123)
        logging.info(f"We sampled {len(df_input)} from the {n_inputs} due to max_instances.")

    optional_fields = {
        "criteria": "",
        "output_price_per_example": np.nan,
        "output_time_per_example": np.nan,
        "category": "",
        "weight": 1.0,
    }
    required_fields = {"instruction", "output", "rubric", "evaluation"}
    for step in ["evaluation", "output"]:
        for col in ["{step}_date", "{step}_time_per_example", "{step}_price_per_example", "{step}_version"]:
            required_fields.add(col.format(step=step))
    required_fields = {r for r in required_fields if r not in optional_fields}
    process_input_df_(df_input, required_fields=required_fields, optional_fields=optional_fields)

    if summarizer_configs:
        summary_completions = summarize(df_input, summarizer_configs, **summarizer_kwargs)
    else:
        summary_completions = None
    report_dict = make_evaluation_report_dict(df_input, summary_completions=summary_completions)
    report_str = format_evaluation_report_md(report_dict)

    return report_dict, report_str


def evaluate_and_generate_report(
    input_path: AnyPath,
    output_path: Optional[AnyPath] = None,
    evaluator_configs: str = "gpt-4o-2024-08-06_CoT_v0",
    summarizer_configs: Optional[str] = "unstructured_gpt-4o-2024-08-06",
    **evaluator_and_summarizer_kwargs,
) -> Path:
    """Evaluate model outputs using rubrics, generate a report, and save it."""
    eval_path = evaluate(input_path, output_path, evaluator_configs, **evaluator_and_summarizer_kwargs)
    report_path = generate_report(eval_path, output_path, summarizer_configs, **evaluator_and_summarizer_kwargs)
    return report_path


def generate_outputs_and_evaluation_report(
    input_path: AnyPath,
    output_path: Optional[AnyPath] = None,
    model_configs: Optional[str] = None,
    evaluator_configs: str = "gpt-4o-2024-08-06_CoT_v0",
    summarizer_configs: Optional[str] = "unstructured_gpt-4o-2024-08-06",
    **completor_evaluator_summarizer_kwargs,
):
    """Generate model outputs, evaluate them using rubrics, and generate the evaluation report. If `model_configs` is
    `None`, then the function will not generate outputs, so the input DataFrame should contain an "output" column.
    """
    completion_path = generate_outputs(
        input_path=input_path,
        output_path=output_path,
        model_configs=model_configs,
        **completor_evaluator_summarizer_kwargs,
    )
    report_path = evaluate_and_generate_report(
        completion_path,
        output_path,
        evaluator_configs,
        **completor_evaluator_summarizer_kwargs,
    )


#
# def get_instructions(
#     n_max_examples: int,
#     category: str | None = None,
#     instruction_set: str = "auto",
#     with_additional_info: bool = False,
#     random_seed: int = 123,
#     output_path: Optional[AnyPath] = None,
#     cache_dir: str = "auto",
# ) -> Optional[pd.DataFrame]:
#     assert instruction_set in ["auto", "alpaca_eval_2", "wildbench-v1"]
#     if instruction_set == "alpaca_eval_2":
#         # TODO(stella): this branch doesn't seem to work: `BuilderConfig 'v1-legacy' not found. Available: ['alpaca_eval', 'alpaca_eval_gpt4_baseline', 'alpaca_eval_all_outputs', 'alpaca_farm_human_annotations', 'alpaca_farm_human_crossannotations', 'alpaca_eval_annotations_alpaca_eval_gpt4', 'alpaca_eval_annotations_claude']`
#         instructions = ae_const.ALPACAEVAL_REFERENCE_OUTPUTS_2()
#         df_instructions = instructions.to_pandas().sample(n_max_examples, random_state=random_seed)
#         df_instructions["instruction"] = df_instructions["instruction"]
#         df_instructions["category"] = df_instructions["dataset"]
#     elif instruction_set == "wildbench-v1":
#         ds = datasets.load_dataset("allenai/WildBench", "v1-legacy")["test"]
#         ds = ds.to_pandas()
#         ds = ds[
#             ds.apply(lambda x: len(x["conversation_input"]) == 1, axis=1)
#         ]  # maintain single-turn instructions for now
#         df_instructions = pd.DataFrame()
#         df_instructions["instruction"] = ds["conversation_input"].apply(lambda x: x[0]["content"])
#         df_instructions["category"] = ds["category"]
#         if category:
#             n_max_examples = min(
#                 n_max_examples,
#                 len(df_instructions[df_instructions["category"] == category]),
#             )
#             df_instructions = df_instructions[df_instructions["category"] == category].sample(
#                 n_max_examples, random_state=random_seed
#             )
#         else:
#             df_instructions = df_instructions.sample(n_max_examples, random_state=random_seed)
#         if with_additional_info:
#             # merge the intent and checklist into the useful_info_to_eval_instruction
#             def _get_useful_info_to_eval_instruction(x):
#                 info = ""
#                 info += f"User intent: {x['intent']}\n"
#                 checklist = "- " + "\n- ".join(x["checklist"].tolist())
#                 info += f"Reference checklist:\n{checklist}\n"
#
#                 return info
#
#             df_instructions["useful_info_to_eval_instruction"] = ds.apply(_get_useful_info_to_eval_instruction, axis=1)
#         else:
#             df_instructions["useful_info_to_eval_instruction"] = "N/A"
#     if output_path is not None:
#         df_instructions.to_json(output_path, orient="records", indent=4)
#     else:
#         return df_instructions

#
# def get_evaluations(outputs, cache_dir: str = "auto", **kwargs):
#     caching_path = cache_dir
#     if cache_dir is not None and cache_dir != "auto":
#         caching_path = Path(cache_dir) / "evaluator_configs.json"
#     evaluator = Evaluator(caching_path=caching_path, **kwargs)
#     scores = evaluator(outputs)
#     df_scores = ae_utils.convert_to_dataframe(scores).query("not output.isin(['', ' '])")
#     df_scores = df_scores.dropna(subset=["criteria_scores"])
#
#     def calc_mean_with_exception_handling(x):
#         x_value_set = set(x.values())
#         if "N/A" in x_value_set or "n/a" in x_value_set:
#             return None
#         else:
#             try:
#                 return pd.Series(x).mean()
#             except:
#                 print(f"x: {x}")
#                 raise
#
#     df_scores["avg_score"] = df_scores["criteria_scores"].apply(calc_mean_with_exception_handling)
#     df_scores = df_scores[df_scores["avg_score"].notnull()]
#
#     def get_criteria_annotation(s):
#         ret = {}
#         for k, v in s["criteria_scores"].items():
#             if v == 0 or v is None:
#                 ret[k] = None
#             else:
#                 ret[k] = dict_reverser(s["scoring_scales"])[v]
#         return ret
#
#     df_criteria_annotations = df_scores.apply(
#         get_criteria_annotation,
#         axis=1,
#     )
#     assert not df_criteria_annotations.empty
#     df_scores["criteria_annotations"] = df_criteria_annotations
#     return df_scores


ALL_FUNCTIONS = {
    "brainstorm_rubrics": brainstorm_rubrics,
    "generate_rubrics": generate_rubrics,
    "generate_outputs": generate_outputs,
    "evaluate": evaluate,
    "generate_report": generate_report,
    "brainstorm_and_generate_rubrics": brainstorm_and_generate_rubrics,
    "evaluate_and_generate_report": evaluate_and_generate_report,
    "generate_outputs_and_evaluation_report": generate_outputs_and_evaluation_report,
}


def main():
    is_fn_name = len(sys.argv) > 1 and "--" not in sys.argv[1]
    is_help = any(a == "--help" for a in sys.argv)

    if is_fn_name or is_help:
        fire.Fire(ALL_FUNCTIONS)
    else:
        # default behavior if no function is specified
        fire.Fire(generate_outputs_and_evaluation_report)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run_doctests":
        run_doctests()
    else:
        fire.Fire(ALL_FUNCTIONS)
