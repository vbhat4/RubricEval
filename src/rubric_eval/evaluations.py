import ast
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from alpaca_eval import utils as ae_utils
from alpaca_eval.annotators import base

from .helpers import CONFIGS_DIR, expand_json_column, mean

__all__ = ["Evaluator", "make_evaluation_report_dict", "format_evaluation_report_md"]


class Evaluator(base.BaseAnnotatorJSON):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto evaluator of the output using the rubric.",
    )
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = object  # use object to have dict output
    DEFAULT_BASE_DIR = CONFIGS_DIR / "evaluators_configs"
    annotator_column = "evaluator"

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = (
            "instruction",
            "criteria",
            "rubric",
            "output",
            "excellent_response",
        ),
        annotators_config="gpt-4o-2024-08-06_CoT_v0",
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
        return "evaluation"

    def make_df_rubric_grading(
        self,
        annotated: Sequence[dict],
    ) -> pd.DataFrame:
        """Add scores to the dataframe based on the rubric."""
        df_eval = ae_utils.convert_to_dataframe(annotated)

        df_eval = df_eval.dropna(subset=[self.annotation_key])

        mask_str = df_eval[self.annotation_key].apply(lambda x: isinstance(x, str))
        if mask_str.any():
            df_eval.loc[mask_str, self.annotation_key] = df_eval.loc[mask_str, self.annotation_key].apply(
                ae_utils.convert_str_to_sequence
            )
            mask_str_new = df_eval[self.annotation_key].apply(lambda x: isinstance(x, str))
            if mask_str_new.any():
                # let's drop this time
                logging.warning(
                    f"{mask_str_new.sum()} examples have string annotations in {self.annotation_key} we are droping them."
                )
                df_eval = df_eval[~mask_str_new]
            else:
                logging.warning(
                    f"{mask_str.sum()} examples had string annotations in {self.annotation_key}, we converted them."
                )

        performance_to_score = dict(excellent=4, good=3, fair=2, poor=1)

        def get_score_for_criterion(d: dict) -> float | int:
            if "likert_score" in d:
                return d["likert_score"]
            else:
                # convert the performance to a score between 1-10
                return (performance_to_score[d["performance"]] - 1) * 3 + 1

        df_eval[self.annotation_key] = df_eval[self.annotation_key].apply(
            lambda x: [{**d, "score": get_score_for_criterion(d)} for d in x]
        )
        # to get the per example score, you take the score for each criterion in self.annotation_key, you then weight
        # them by the weight in "rubric" and sum them up
        df_eval["unweighted_score"] = df_eval[self.annotation_key].apply(lambda x: mean([d["score"] for d in x]))

        df_eval["weighted_score"] = df_eval.apply(_compute_score_from_rubric_and_grading, axis=1)
        if df_eval["weighted_score"].isnull().any():
            n_scores_missing = df_eval["weighted_score"].isnull().sum()
            logging.warning(
                f"{n_scores_missing} examples have missing scores. Probably because the criteria don't have the same names."
            )

        # new
        df_eval["unweighted_score"] = np.where(df_eval["output"] == "", np.nan, df_eval["unweighted_score"])
        df_eval["weighted_score"] = np.where(df_eval["output"] == "", np.nan, df_eval["weighted_score"])

        return df_eval


def make_evaluation_report_dict(
    df: pd.DataFrame, summary_completions: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Generate a report dict from the evaluation DataFrame.

    Args:
        df (pd.DataFrame): The evaluation DataFrame. Should have columns "model", "evaluator", "dataset_name",
            "output_date", "evaluation_date", "evaluator_version", "output_time_per_example", "output_price_per_example",
            "evaluation_time_per_example", "evaluation_price_per_example", "weighted_score", "unweighted_score", "category".
        summary_completions (bool): Optional summary of the evaluations.
    """

    def date_only(x: str) -> str:
        return datetime.fromisoformat(x).strftime("%Y-%m-%d")

    n_instructions = len(df)
    eval_time_per_eg = df["evaluation_time_per_example"].mean()
    eval_price_per_eg = df["evaluation_price_per_example"].mean()

    report_dict = dict(
        # --- benchmark details ---
        model=_get_unique_val_from_col(df, "model"),
        evaluator=_get_unique_val_from_col(df, "evaluator"),
        dataset=_get_unique_val_from_col(df, "dataset_name"),
        n_instructions=n_instructions,
        n_categories=len(df["category"].unique()),
        report_date=date_only(datetime.now().isoformat()),
        output_date=_get_unique_val_from_col(df, "output_date", transform=date_only),
        evaluation_date=_get_unique_val_from_col(df, "evaluation_date", transform=date_only),
        evaluator_version=_get_unique_val_from_col(df, "evaluation_version"),
        report_version=ae_utils.get_multi_package_version(["rubric_eval"]),
        # --- output details ---
        avg_n_chars=df["output"].apply(len).mean(),
        avg_n_list_items=df["output"].apply(ae_utils.contains_list, is_return_count=True).mean(),
        proba_list=df["output"].apply(ae_utils.contains_list, is_return_count=False).mean(),
        # --- time & cost ---
        evaluation_time_per_example=eval_time_per_eg,
        evaluation_price_per_example=eval_price_per_eg,
        output_time_per_example=df["output_time_per_example"].mean(),
        output_price_per_example=df["output_price_per_example"].mean(),
        # report_cost=n_instructions * (eval_price_per_eg),
        # report_time=n_instructions * (eval_time_per_eg),
        # --- quantitative ---
        weighted_score_mean=df["weighted_score"].mean(),
        weighted_score_sem=df["weighted_score"].sem(),
        unweighted_score_mean=df["unweighted_score"].mean(),
        unweighted_score_sem=df["unweighted_score"].sem(),
        weighted_score_per_category=df.groupby("category")["weighted_score"].mean().to_dict(),
        weighted_sem_per_category=df.groupby("category")["weighted_score"].sem().to_dict(),
        unweighted_score_per_category=df.groupby("category")["unweighted_score"].mean().to_dict(),
        unweighted_sem_per_category=df.groupby("category")["unweighted_score"].sem().to_dict(),
        # score_per_cagtegory
        # --- qualitative ---
        # pro_feedback=["feedback 1", "feedback 2"],
        # con_feedback=["feedback 3", "feedback 4"],
        # feedback_per_category
    )

    if summary_completions is not None:
        # run the summarizer
        summary_keys_to_keep = [
            k for k in summary_completions.keys() if k.startswith("summar") and k not in ["summarizer_completions_all"]
        ]
        report_dict.update(**{k: v for k, v in summary_completions.items() if k in summary_keys_to_keep})
        report_cost = n_instructions * (eval_price_per_eg + summary_completions["summarizer_price_per_example"])
        report_time = n_instructions * (eval_time_per_eg + summary_completions["summarizer_time_per_example"])
    else:
        report_cost = n_instructions * eval_price_per_eg
        report_time = n_instructions * eval_time_per_eg

    report_dict["report_cost"] = report_cost
    report_dict["report_time"] = report_time

    return report_dict


def format_evaluation_report_md(report_dict: dict[str, Any]) -> str:
    rd = report_dict
    is_category = rd["n_categories"] > 1

    qualitative_str = ""
    overview_str_sffx = ""
    quantitative_str_per_category = ""

    if "summary" in rd:
        strengths_formatter = lambda d, prfx="": prfx + "+ " + ("\n" + prfx + "+ ").join(d["summary"]["strengths"])
        weaknesses_formatter = lambda d, prfx="": prfx + "- " + ("\n" + prfx + "- ").join(d["summary"]["weaknesses"])
        suggestions_formatter = lambda d, prfx="": prfx + "- " + ("\n" + prfx + "- ").join(d["summary"]["suggestions"])

        strength_str = strengths_formatter(rd)
        weaknesses_str = weaknesses_formatter(rd)
        suggestions_str = suggestions_formatter(rd)

        overview_str_sffx += f"""
**Pro Feedback**:
{strength_str}

**Con Feedback**:
{weaknesses_str}
"""

        qualitative_str += f"""
## Qualitative
**Suggestions**:
{suggestions_str}

**Overal assessment**:
{rd["summary"]["overall_assessment"]}
"""
        if is_category:
            prfx = "\t\t"
            feedback_per_category_str = "\n".join(
                [
                    f"""* **{k}**:
    * **Strengths**:
{strengths_formatter(v, prfx=prfx)}
    * **Weaknesses**:
{weaknesses_formatter(v, prfx=prfx)}"""
                    for k, v in rd["summaries_splitted"].items()
                ]
            )

            qualitative_str += f"""
**Qualitative feedback per category**:
{feedback_per_category_str}
"""

    if is_category:
        weighted_score_per_category_str = "\n".join(
            [
                f"- {k}: {v:.2f} ± {rd['weighted_sem_per_category'][k]:.2f}"
                for k, v in rd["weighted_score_per_category"].items()
            ]
        )

        quantitative_str_per_category += f"""
## Quantitative score per category
{weighted_score_per_category_str}
"""

    return f"""# Evaluation report for model="{rd["model"]}" on dataset="{rd["dataset"]}"

## Overview
**Score**: {rd["weighted_score_mean"]:.2f} ± {rd["weighted_score_sem"]:.2f}
{overview_str_sffx}
## Details
**Model**: {rd["model"]}

**Dataset**: {rd["dataset"]}

**N Instructions**: {rd["n_instructions"]}

**Evaluator**: {rd["evaluator"]}

**Summarizer**: {rd["summarizer"] if "summarizer" in rd else "None"}

**Report Date**: {rd["report_date"]}

**Version**: {rd["report_version"]}

**Report Cost**: ${rd["report_cost"]:.2f}

**Report Time**: {rd["report_time"]/60:.2f} minutes

## Analysis of outputs
**Avg. length of output**: {rd["avg_n_chars"]:.0f} characters

**Avg. list presence**: {rd["proba_list"]*100:.1f} %

**Avg. number of list items**: {rd["avg_n_list_items"]:.1f} list items

{qualitative_str}
"""


### Helpers


def _compute_score_from_rubric_and_grading(x: dict) -> float:
    """Compute the score from the rubric and grading."""
    df_grading = pd.DataFrame(x["evaluation"])
    if "criterion" not in df_grading.columns:
        return np.nan
    df_grading = df_grading.set_index("criterion")
    df_rubric = pd.DataFrame(x["rubric"]).set_index("criterion")
    necessary_indices = df_rubric.query("weight > 0").index
    if not (set(necessary_indices).issubset(set(df_grading.index))):
        return np.nan
    out = (df_grading["score"] * df_rubric.reindex(df_grading.index)["weight"] / 100).sum()
    return out


def _get_unique_val_from_col(
    df: pd.DataFrame,
    col: str,
    transform: Optional[Callable] = None,
) -> str:
    """Get the unique value from a column in a DataFrame."""
    if col not in df:
        return "<Unknown>"
    series = df[col]
    if transform is not None:
        series = series.apply(transform)
    unique = series.unique()
    if len(unique) == 1:
        if unique is None:
            out = "<Unknown>"
        else:
            out = unique[0]
    else:
        out = "<Multiple>"
    return out
