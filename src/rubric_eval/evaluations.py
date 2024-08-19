import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from alpaca_eval import utils as ae_utils
from alpaca_eval.annotators import base

from .helper import CONFIGS_DIR, expand_json_column, mean

__all__ = ["Evaluator"]


class Evaluator(base.BaseAnnotatorJSON):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto evaluator of the output using the rubric.",
    )
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = "object"  # use object to have dict output
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
        return "rubric_grading"

    def make_df_rubric_grading(
        self,
        annotated: Sequence[dict],
    ) -> pd.DataFrame:
        """Add scores to the dataframe based on the rubric."""
        df_eval = ae_utils.convert_to_dataframe(annotated)
        performance_to_score = dict(excellent=4, good=3, fair=2, poor=1)
        df_eval[self.annotation_key] = df_eval[self.annotation_key].apply(
            # add "score" based on "performance" mapped by dict(excellent=4, good=3, fair=2, poor=1)
            lambda x: [{**d, "score": performance_to_score[d["performance"]]} for d in x]
        )
        # to get the per example score, you take the score for each criterion in self.annotation_key, you then weight
        # them by the weight in "rubric" and sum them up
        df_eval["unweighted_score"] = df_eval[self.annotation_key].apply(lambda x: mean([d["score"] for d in x]))
        df_eval["weighted_score"] = df_eval.apply(compute_score_from_rubric_and_grading, axis=1)
        if df_eval["weighted_score"].isnull().any():
            n_scores_missing = df_eval["weighted_score"].isnull().sum()
            logging.warning(
                f"{n_scores_missing} examples have missing scores. Probably because the criteria don't have the same names."
            )
        return df_eval


def compute_score_from_rubric_and_grading(x: dict) -> float:
    """Compute the score from the rubric and grading."""
    df_grading = pd.DataFrame(x["rubric_grading"]).set_index("criterion")
    df_rubric = pd.DataFrame(x["rubric"]).set_index("criterion")
    if not df_grading.index.equals(df_rubric.index):
        return np.nan
    out = (df_grading["score"] * df_rubric.reindex(df_grading.index)["weight"] / 100).sum()
    return out
