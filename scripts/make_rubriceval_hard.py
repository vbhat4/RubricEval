from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from alpaca_eval import utils as ae_utils
from alpaca_eval.annotators import base
from rubric_eval import helpers as re_helpers
from rubric_eval.main import brainstorm_and_generate_rubrics, generate_outputs


class Differentiator(base.BaseAnnotatorJSON):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto annotator for annotating the difference between model outputs.",
    )
    DEFAULT_ANNOTATION_TYPE = str
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_BASE_DIR = Path(__file__).parent / "rubric_brainstormers_configs"
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
    outputs_main_path, outputs_alt_path, save_path=None, max_samples: Optional[int] = None, **decoding_kwargs
):
    """
    This function is used to annotate how different two answers are from each other.
    """
    df_main = pd.read_json(outputs_main_path)
    df_alt = pd.read_json(outputs_alt_path)
    if max_samples:
        df_main = df_main.iloc[:max_samples]
        df_alt = df_alt.iloc[:max_samples]
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


def make_rubriceval_hard(instructions_path: Path, n_cat: int = 5, n_per_cat: int = 10):
    diff_path = re_helpers.MAIN_DIR / "data/help/rubriceval_general/differentiator.json"
    benchmark_path = re_helpers.MAIN_DIR / "data/benchmark/rubriceval_general/benchmark.json"
    main_output_path = re_helpers.MAIN_DIR / "results/gpt-4o-2024-08-06/rubriceval_general/outputs.json"
    alt_output_path = re_helpers.MAIN_DIR / "results/gpt-4o-mini-2024-07-18/rubriceval_general/outputs.json"
    # Generate outputs for gpt-4o-2024-08-06
    generate_outputs(
        model_configs="gpt-4o-2024-08-06",
        input_path=benchmark_path,
        output_path=main_output_path,
        is_rm_prev_columns=False,
    )
    # Generate outputs for gpt-4o-mini-2024-07-18
    generate_outputs(
        model_configs="gpt-4o-mini-2024-07-18",
        input_path=benchmark_path,
        output_path=alt_output_path,
        is_rm_prev_columns=False,
    )
    annotate_diff_outputs(main_output_path, alt_output_path, save_path=diff_path)
    df = pd.read_json(diff_path)
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
    df_groups = df_groups.query("count >= 10").sort_values("mean", ascending=False)
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
    instructions_path.parent.mkdir(parents=True, exist_ok=True)
    cols_to_drop = list(df_hard.filter(like="annotations_", axis=1).columns)
    cols_to_drop += ["mean_score", "differentiator", "output_alt", "output_main", "mean_score"]
    df_hard.drop(columns=cols_to_drop, inplace=True)
    df_hard.to_json(instructions_path, orient="records", indent=2)


if __name__ == "__main__":
    instructions_path = re_helpers.MAIN_DIR / "data/benchmark/rubriceval_hard/instructions.json"
    make_rubriceval_hard(instructions_path)
    rubric_path = re_helpers.MAIN_DIR / "data/benchmark/rubriceval_hard/benchmark.json"
    brainstorm_and_generate_rubrics(input_path=instructions_path, output_path=rubric_path)
