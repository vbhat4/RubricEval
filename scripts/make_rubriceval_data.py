import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import fire
import pandas as pd
from alpaca_eval import utils as ae_utils
from alpaca_eval.annotators import base
from experiments.evaluators import Checklister, ListRubricator

from rubric_eval import helpers as re_helpers
from rubric_eval.main import (
    brainstorm_and_generate_rubrics,
    brainstorm_rubrics_from_df,
    generate_outputs,
    generate_rubrics_from_df,
)
from rubric_eval.rubrics import Rubricator, RubricBrainstormer


class Differentiator(base.BaseAnnotatorJSON):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto annotator for annotating the difference between model outputs.",
    )
    DEFAULT_ANNOTATION_TYPE = str
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_BASE_DIR = Path(__file__).parent / "differentiator_configs"
    annotator_column = "differentiator"

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = ("instruction", "output_main", "output_alt"),
        annotators_config="gpt-4o-2024-08-06_v0",
        **kwargs,
    ):
        super().__init__(
            *args,
            annotators_config=annotators_config,
            primary_keys=primary_keys,
            packages_for_which_to_show_version=["rubric_eval", "alpaca_eval"],
            **kwargs,
        )

    @property
    def annotation_key(self) -> str:
        return "annotations"

    def annotate_head2head(
        self,
        outputs_main: pd.DataFrame,
        outputs_alt: pd.DataFrame,
        keys_to_merge: Optional[Sequence[str]] = ("instruction",),
        **decoding_kwargs,
    ):
        keys_to_merge = list(keys_to_merge)

        outputs_main = outputs_main.copy()
        outputs_alt = outputs_alt.copy()
        outputs_main["tmp_idx"] = range(len(outputs_main))
        outputs_alt["tmp_idx"] = range(len(outputs_main))
        keys_to_merge += ["tmp_idx"]  # add a temporary index to merge on

        # find all the columns that are in both
        other_same_cols = [
            k for k in outputs_main.columns if k in outputs_alt and k not in (keys_to_merge + ["output"])
        ]

        df_to_annotate = pd.merge(
            outputs_main,
            outputs_alt,
            on=keys_to_merge,
            suffixes=("_main", "_alt"),
        )

        for c in other_same_cols:
            # if the columns are the same, we can drop the _2. but dont' skip for generator and output
            if c not in ["output", "model"] and df_to_annotate[c + "_main"].equals(df_to_annotate[c + "_alt"]):
                df_to_annotate = df_to_annotate.drop(columns=c + "_alt").rename(columns={c + "_main": c})

        df_to_annotate = df_to_annotate.drop(columns="tmp_idx")

        out = self.__call__(df_to_annotate, **decoding_kwargs)

        return out


def annotate_diff_outputs(
    outputs_main_path, outputs_alt_path, save_path=None, max_instances: Optional[int] = None, **decoding_kwargs
):
    """
    This function is used to annotate how different two answers are from each other.
    """
    df_main = pd.read_json(outputs_main_path)
    df_alt = pd.read_json(outputs_alt_path)
    if max_instances:
        df_main = df_main.iloc[:max_instances]
        df_alt = df_alt.iloc[:max_instances]
    keys_to_merge = ["instruction", "id"]
    cols_to_keep = ["output", "category", "useful_info_to_eval_instruction"] + keys_to_merge
    diffentiator = Differentiator()
    annotated = diffentiator.annotate_head2head(
        df_main[cols_to_keep], df_alt[cols_to_keep], keys_to_merge=keys_to_merge, **decoding_kwargs
    )
    df = ae_utils.convert_to_dataframe(annotated)
    df = re_helpers.expand_json_column(df, diffentiator.annotation_key)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(save_path, orient="records", indent=2)


def get_differentiated_data(
    skip_diff_if_exists: bool = True,
    max_instances: Optional[int] = None,
):
    diff_path = re_helpers.MAIN_DIR / "data/help/rubriceval_general/differentiator.json"
    if skip_diff_if_exists and diff_path.exists():
        logging.info(f"Skipping differentiator because {diff_path} already exists")
    else:
        benchmark_path = re_helpers.MAIN_DIR / "data/benchmark/rubriceval_general/benchmark.json"
        main_output_path = re_helpers.MAIN_DIR / "results/gpt-4o-2024-08-06/rubriceval_general/outputs.json"
        alt_output_path = re_helpers.MAIN_DIR / "results/gpt-4o-mini-2024-07-18/rubriceval_general/outputs.json"
        # Generate outputs for gpt-4o-2024-08-06
        generate_outputs(
            model_configs="gpt-4o-2024-08-06",
            input_path=benchmark_path,
            output_path=main_output_path,
            is_rm_prev_columns=False,
            max_instances=max_instances,
        )
        # Generate outputs for gpt-4o-mini-2024-07-18
        generate_outputs(
            model_configs="gpt-4o-mini-2024-07-18",
            input_path=benchmark_path,
            output_path=alt_output_path,
            is_rm_prev_columns=False,
            max_instances=max_instances,
        )
        annotate_diff_outputs(main_output_path, alt_output_path, save_path=diff_path, max_instances=max_instances)
    df = pd.read_json(diff_path)
    if max_instances:
        df = df.iloc[:max_instances]
    return df


def save_rubriceval_data(df: pd.DataFrame, instructions_path: Path):
    instructions_path.parent.mkdir(parents=True, exist_ok=True)
    cols_to_drop = list(df.filter(like="annotations_", axis=1).columns)
    cols_to_drop += ["mean_score", "differentiator", "output_alt", "output_main"]
    # remove user_intent for our experiments as we want all done by the same model
    cols_to_drop += ["useful_info_to_eval_instruction"]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    df.to_json(instructions_path, orient="records", indent=2)


def make_rubriceval_hard(
    instructions_path: Path,
    n_cat: int = 8,
    n_per_cat: int = 25,
    max_instances: Optional[int] = None,
    is_show_counts: bool = False,
):
    df = get_differentiated_data(max_instances=max_instances)
    df_groups = pd.merge(
        df.groupby("category")["difference_score"].mean(),
        df.groupby("category")["complexity_score"].mean(),
        on="category",
    )
    df_groups = pd.merge(
        df_groups,
        df.groupby("category")["objective_score"].mean(),
        on="category",
    )
    df_groups["mean"] = df_groups.mean(axis=1)
    # add the count of samples per category
    counts = df.groupby("category").size()
    counts.name = "count"
    df_groups = pd.merge(df_groups, counts, on="category")
    df_groups = df_groups.query("count >= @n_per_cat").sort_values("mean", ascending=False)
    categories_to_include = df_groups.head(n_cat).index.tolist()
    df_filtered = df[df["category"].isin(categories_to_include)].copy()
    df_filtered["mean_score"] = (
        df_filtered["objective_score"] + df_filtered["difference_score"] + df_filtered["complexity_score"]
    ) / 3
    # select the examples that have the max n_per_cat `mean_score` per category
    df_hard = (
        df_filtered.groupby("category")
        .apply(lambda x: x.sort_values("mean_score", ascending=False).head(n_per_cat))
        .reset_index(drop=True)
    )
    if is_show_counts:
        print(df_hard["category"].value_counts())
    save_rubriceval_data(df_hard, instructions_path)


def make_rubriceval_sampled(
    instructions_path: Path,
    n_per_cat: int = 10,  # 25,
    max_instances: Optional[int] = None,
    is_show_counts: bool = False,
):
    df = get_differentiated_data(max_instances=max_instances)
    counts = df.groupby("category").size()
    counts.name = "count"
    df_groups = pd.merge(df, counts, on="category")
    # df_groups = df_groups.query("count >= @n_per_cat")
    df_groups = df_groups.query("count >= 25")  # TODO uncomment above
    # sample n_per_cat using shuffle so that you can hit the cache if incresing n_per_cat
    df_sampled = (
        df_groups.groupby("category")
        .apply(lambda x: x.sample(frac=1, random_state=123))  # Shuffle within each category
        .groupby(level="category")
        .head(n_per_cat)  # Take top n_per_cat from each category
        .reset_index(drop=True)
    )
    if is_show_counts:
        print(df_sampled["category"].value_counts())
    save_rubriceval_data(df_sampled, instructions_path)


def rubriceval_hard(max_instances: Optional[int] = None, **kwargs):
    instructions_path = re_helpers.MAIN_DIR / "data/benchmark/rubriceval_hard/instructions.json"
    make_rubriceval_hard(instructions_path, **kwargs)
    rubric_path = re_helpers.MAIN_DIR / "data/benchmark/rubriceval_hard/benchmark.json"
    brainstorm_and_generate_rubrics(input_path=instructions_path, output_path=rubric_path, max_instances=max_instances)


def rubriceval_sampled(max_instances: Optional[int] = None, **kwargs):
    instructions_path = re_helpers.MAIN_DIR / "data/benchmark/rubriceval_sampled/instructions.json"
    make_rubriceval_sampled(instructions_path, **kwargs)
    rubric_path = re_helpers.MAIN_DIR / "data/benchmark/rubriceval_sampled/benchmark.json"
    brainstorm_and_generate_rubrics(input_path=instructions_path, output_path=rubric_path, max_instances=max_instances)


def rubriceval_stanford_ml(is_expert_annotated: bool = True, is_rubric **kwargs):
    readable_dataset_path = Path("data/benchmark/rubriceval_stanford_ml/readable_dataset")
    readable_dataset_path.mkdir(parents=True, exist_ok=True)
    instructions_path = (
        re_helpers.MAIN_DIR / "data/benchmark/rubriceval_stanford_ml/rubriceval_stanford_ml_instructions.json"
    )
    if is_expert_annotated:

    instructions_path = (
        re_helpers.MAIN_DIR / "data/benchmark/rubriceval_stanford_ml/rubriceval_stanford_ml_instructions.json"
    )
    # data is already written
    df = pd.read_json(instructions_path)

    annotators = []
    ann_kwargs = dict(other_output_keys_to_keep=[], **kwargs)

    # add columns from auto generated checklist
    checklister = Checklister(**ann_kwargs)
    df = ae_utils.convert_to_dataframe(checklister(df))
    annotators.append(checklister)

    # add columns from auto generated list rubric
    list_rubricator = ListRubricator(**ann_kwargs)
    df = ae_utils.convert_to_dataframe(list_rubricator(df))
    df = re_helpers.expand_json_column(df, list_rubricator.annotation_key)
    df.drop(columns=["strong_response"], inplace=True)
    annotators.append(list_rubricator)

    # add columns from auto brainstormed rubrics
    if "brainstormed_rubric" not in df.columns:
        rubric_brainstormer = RubricBrainstormer(**ann_kwargs)
        df["useful_info_to_eval_instruction"] = "Here's a good solution to the problem:\n" + df["expert_solution"]
        df = rubric_brainstormer.make_df_rubrics(
            rubric_brainstormer(df), is_renormalize_weight=True, is_extract_criteria_col=False
        )
        df.drop(
            columns=["brainstormed_response", "useful_info_to_eval_instruction", "learning_objectives"], inplace=True
        )
        annotators.append(rubric_brainstormer)

    else:
        # add columns from auto generated rubrics
        rubricator = Rubricator(**ann_kwargs)
        df["brainstormed_response"] = df["expert_solution"]
        df = rubricator.make_df_rubrics(rubricator(df), is_extract_criteria_col=True, is_renormalize_weight=True)
        annotators.append(rubricator)

    for ann in annotators:
        df[f"expert_{ann.annotation_key}"] = df[ann.annotation_key]
        df[f"expert_{ann.annotation_key}_time_sec"] = ""
        df.drop(columns=[ann.annotator_column, ann.annotation_key], inplace=True)

    for i, row in df.iterrows():
        readable_dataset = ""
        for k, v in row.to_dict().items():
            if k == "short_name":
                continue
            if isinstance(v, (list, tuple, dict)):
                v = json.dumps(v, indent=2)
            readable_dataset += f"# <{k}>:\n{v}\n"
        # save to markdown file
        with open(readable_dataset_path / f"{row['short_name']}.md", "w") as file:
            file.write(readable_dataset)


if __name__ == "__main__":
    fire.Fire({"hard": rubriceval_hard, "sampled": rubriceval_sampled, "stanford_ml": rubriceval_stanford_ml})
