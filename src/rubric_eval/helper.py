from rubric_eval import Instructionator, Completor, Evaluator, RubricGenerator
from typing import List, Dict, Any

import datasets
from alpaca_eval import utils as ae_utils
from alpaca_eval import constants as ae_const
import ast
import pandas as pd
from pathlib import Path
from alpaca_eval.types import AnyPath
from typing import Optional


def dict_reverser(d):
    return {v: k for k, v in d.items()}


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
    if instruction_set == "auto":
        caching_path = cache_dir
        if cache_dir is not None and cache_dir != "auto":
            caching_path = Path(cache_dir) / "instructionator_configs.json"
        instructionator = Instructionator(caching_path=caching_path)
        assert category is not None, "category must be provided if instruction_set is 'auto'"
        output = instructionator.generate_n_instruction_for_a_category(
            n=n_max_examples, category=category
        )
        instructions = ast.literal_eval(output[0]["categories_and_instructions"])
        df_instructions = ae_utils.convert_to_dataframe(instructions)
    elif instruction_set == "alpaca_eval_2":
        # TODO(stella): this branch doesn't seem to work: `BuilderConfig 'v1-legacy' not found. Available: ['alpaca_eval', 'alpaca_eval_gpt4_baseline', 'alpaca_eval_all_outputs', 'alpaca_farm_human_annotations', 'alpaca_farm_human_crossannotations', 'alpaca_eval_annotations_alpaca_eval_gpt4', 'alpaca_eval_annotations_claude']`
        instructions = ae_const.ALPACAEVAL_REFERENCE_OUTPUTS_2()
        df_instructions = instructions.to_pandas().sample(n_max_examples, random_state=random_seed)
        df_instructions["prompt"] = df_instructions["instruction"]
        df_instructions["category"] = df_instructions["dataset"]
    elif instruction_set == "wildbench-v1":
        ds = datasets.load_dataset("allenai/WildBench", 'v1-legacy')["test"]
        ds = ds.to_pandas()
        ds = ds[ds.apply(lambda x: len(x["conversation_input"]) == 1, axis=1)]  # maintain single-turn instructions for now
        df_instructions = pd.DataFrame()
        df_instructions["prompt"] = ds["conversation_input"].apply(lambda x: x[0]["content"])
        df_instructions["category"] = ds["primary_tag"]
        if category:
            n_max_examples = min(n_max_examples, len(df_instructions[df_instructions["category"] == category]))
            df_instructions = df_instructions[df_instructions["category"] == category].sample(n_max_examples, random_state=random_seed)
        else:
            df_instructions = df_instructions.sample(n_max_examples, random_state=random_seed)
        if with_additional_info:
            # merge the intent and checklist into the additional_information
            def _get_additional_information(x):
                info = ""
                info += f"User intent: {x['intent']}\n"
                checklist = '- ' + '\n- '.join(x['checklist'].tolist())
                info += f"Reference checklist:\n{checklist}\n"

                return info
            df_instructions["additional_information"] = ds.apply(_get_additional_information, axis=1)
        else:
            df_instructions["additional_information"] = "N/A"
    if output_path is not None:
        df_instructions.to_json(output_path, orient='records', indent=4)
    else:
        return df_instructions


def get_detailed_rubrics(criteria, cache_dir: str = "auto", **annot_kwargs) -> pd.DataFrame:
    df_criteria = ae_utils.convert_to_dataframe(criteria)
    caching_path = cache_dir
    if cache_dir is not None and cache_dir != "auto":
        caching_path = Path(cache_dir) / "rubric_generator_configs.json"
    rubric_generator = RubricGenerator(caching_path=caching_path, **annot_kwargs)
    detailed_rubrics = rubric_generator(df_criteria)
    df_detailed_rubrics = rubric_generator.make_df_rubrics(detailed_rubrics)
    return df_detailed_rubrics


def get_model_completions(rubrics, model_name: str, cache_dir: str = "auto", **kwargs):
    df_rubrics = ae_utils.convert_to_dataframe(rubrics)
    caching_path = cache_dir
    if cache_dir is not None and cache_dir != "auto":
        caching_path = Path(cache_dir) / "completor_configs.json"
    model = Completor(annotators_config=model_name, caching_path=caching_path, **kwargs)
    completions = pd.DataFrame(model(df_rubrics))
    completions = completions[completions["raw_completion"].notnull()].to_dict(
        orient="records"
    )
    return completions


def get_evaluations(completions, cache_dir: str = "auto", **kwargs):
    caching_path = cache_dir
    if cache_dir is not None and cache_dir != "auto":
        caching_path = Path(cache_dir) / "evaluator_configs.json"
    evaluator = Evaluator(caching_path=caching_path, **kwargs)
    scores = evaluator(completions)
    df_scores = ae_utils.convert_to_dataframe(scores).query(
        "not output.isin(['', ' '])"
    )
    df_scores = df_scores.dropna(subset=['criteria_scores'])

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
    df_scores = df_scores[df_scores['avg_score'].notnull()]

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
