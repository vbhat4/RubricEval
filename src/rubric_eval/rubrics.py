import logging
from typing import Optional, Sequence

import pandas as pd
from alpaca_eval import utils as ae_utils
from alpaca_eval.annotators import base

from .helpers import CONFIGS_DIR, expand_json_column, percentage_scale_df_

__all__ = ["RubricBrainstormer", "Rubricator"]


class BaseRubricator(base.BaseAnnotatorJSON):
    DEFAULT_ANNOTATION_TYPE = object
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = ("instruction",),
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
    def dflt_annotator_kwargs(self):
        # TODO: remove is_log_first_prompt for release
        return dict(available_fields_to_format=self.available_fields_to_format, is_log_first_prompt=True)

    def make_df_rubrics(
        self,
        annotated: Sequence[dict],
        rubric_columns: Optional[Sequence[str]] = None,
        is_renormalize_weight: bool = True,
        is_extract_criteria_col: bool = True,
        is_expand_json_column: bool = False,
    ) -> pd.DataFrame:
        """Processes the annotated examples into a DataFrame containing a rubric of (dict of dict) column."""
        df_rubrics = ae_utils.convert_to_dataframe(annotated)
        n_examples = len(df_rubrics)
        # filter out examples where annotation is missing (pd.isnull(value) or value is None)
        df_rubrics = df_rubrics.dropna(subset=[self.annotation_key])
        n_missing = n_examples - len(df_rubrics)

        if n_missing > 0:
            logging.warning(f"{n_missing} examples have missing annotations for {self.__class__.__name__}.")

        # TODO: will likely end up removing this
        if is_expand_json_column:
            # TODO: understand why this can add back rows that were previously removed
            df_rubrics = expand_json_column(df_rubrics, self.annotation_key)
            df_rubrics = df_rubrics.dropna(subset=[self.annotation_key])

        if rubric_columns is None:
            rubric_columns = [self.annotation_key]

        # TODO: remove hard coding
        if self.annotation_key == "brainstormed_rubric":
            df_rubrics["learning_objectives"] = df_rubrics["brainstormed_rubric"].apply(lambda x: x["learning_objectives"])
            df_rubrics["brainstormed_response"] = df_rubrics["brainstormed_rubric"].apply(lambda x: x["brainstormed_response"])
            df_rubrics["brainstormed_rubric"] = df_rubrics["brainstormed_rubric"].apply(lambda x: x["brainstormed_rubric"])
        elif self.annotation_key == "rubric":
            df_rubrics["rubric"] = df_rubrics["rubric"].apply(lambda x: x["rubric"])

        for col in rubric_columns:
            # remove potential duplicate criteria in the rubric (happens with bad models)
            df_rubrics[col] = df_rubrics[col].apply(
                lambda x: pd.DataFrame(x).drop_duplicates(subset=["criterion"], keep="first").to_dict(orient="records")
            )

            if is_renormalize_weight:
                # each element is a list of dicts, dicts have a key "weight" that we want to normalize over the list
                # we do so by converting list of dict to df, then normalize "weight" then convert back to list of dict
                df_rubrics[col] = df_rubrics[col].apply(
                    lambda x: percentage_scale_df_(pd.DataFrame(x), columns=["weight"]).to_dict(orient="records")
                )

            if is_extract_criteria_col:
                # for each rubric (which is a list of dict) make a list of the key "criterion"
                df_rubrics["criteria"] = df_rubrics[col].apply(lambda x: [c["criterion"] for c in x])

        return df_rubrics


class RubricBrainstormer(BaseRubricator):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto annotator for brainstorming criteria.",
    )
    DEFAULT_BASE_DIR = CONFIGS_DIR / "rubric_brainstormers_configs"
    annotator_column = "brainstormer"

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = ("instruction", "useful_info_to_eval_instruction"),
        annotators_config="gpt-4o-2024-08-06_CoT_v1",
        **kwargs,
    ):
        super().__init__(
            *args,
            annotators_config=annotators_config,
            primary_keys=primary_keys,
            **kwargs,
        )

    @property
    def annotation_key(self) -> str:
        return "brainstormed_rubric"


class Rubricator(BaseRubricator):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto annotator for generating detailed rubrics.",
    )
    DEFAULT_BASE_DIR = CONFIGS_DIR / "rubricators_configs"
    annotator_column = "rubricator"

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = ("instruction", "brainstormed_rubric", "brainstormed_response"),
        annotators_config="gpt-4o-2024-08-06_CoT_v1",
        **kwargs,
    ):
        super().__init__(
            *args,
            annotators_config=annotators_config,
            primary_keys=primary_keys,
            **kwargs,
        )

    @property
    def annotation_key(self) -> str:
        return "rubric"
