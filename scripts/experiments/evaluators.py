"""
File containing the 3 evaluators that are not rubric based:
- NaiveEvaluator # no conditioning
- ChecklistEvaluator # conditioning on a checklist of the rubric
- SolutionEvaluator # conditioning on the gold solution
"""
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
from alpaca_eval import utils as ae_utils
from alpaca_eval.annotators import base

from rubric_eval import helpers, main
from rubric_eval.evaluations import Evaluator as REEvaluator
from rubric_eval.outputs import Outputer
from rubric_eval.rubrics import Rubricator

__all__ = [
    "NaiveEvaluator",
    "ChecklistEvaluator",
    "SolutionEvaluator",
    "RubricEvaluator",
    "BaseEvaluator",
    "RubricSolutionEvaluator",
    "ListRubricEvaluator",
    "ListRubricSolutionEvaluator",
]

SCRIPTS_DIR = Path(__file__).parents[1]
CONFIGS_DIR = SCRIPTS_DIR / "configs"


class BaseEvaluator(base.BaseAnnotatorJSON):
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = object  # use object to have dict output
    annotator_column = "evaluator"
    DEFAULT_BASE_DIR = ...
    PRIMARY_KEYS = ...
    PREPROCESSING_COLUMNS = ...
    PREPROCESSOR_COLUMNS = ...

    def __init__(
        self,
        *args,
        annotators_config: str = "gpt-4o-2024-08-06_CoT_v1",
        **kwargs,
    ):
        super().__init__(
            *args,
            annotators_config=annotators_config,
            primary_keys=self.PRIMARY_KEYS,
            packages_for_which_to_show_version=["rubric_eval", "alpaca_eval"],
            **kwargs,
        )

    @property
    def annotation_key(self) -> str:
        return "evaluation"

    def postprocess(self, df: pd.DataFrame, is_expand_json_column: bool = True) -> pd.DataFrame:
        if is_expand_json_column:
            df = helpers.expand_json_column(df, self.annotation_key)
        df["evaluator_type"] = self.__class__.__name__
        df["evaluator_seed"] = self.seed
        return df

    @classmethod
    def preprocess(cls, df: pd.DataFrame, gold_model: str, is_skip_preprocess_if_exists: bool = False) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def _is_skip_preprocess(cls, df: pd.DataFrame, annotation_key: str, is_skip_preprocess_if_exists: bool) -> bool:
        is_skip = annotation_key in df.columns and df[annotation_key].notna().all() and is_skip_preprocess_if_exists
        if is_skip:
            logging.info(f"Skipping preprocess for {cls.__name__} because {annotation_key} is already present")
        return is_skip


class NaiveEvaluator(BaseEvaluator):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto evaluator of the output using no conditioning.",
    )
    DEFAULT_BASE_DIR = CONFIGS_DIR / "naive_evaluators_configs"
    PRIMARY_KEYS = (
        "instruction",
        "output",
    )
    PREPROCESSING_COLUMNS = []
    PREPROCESSOR_COLUMNS = []

    @classmethod
    def preprocess(cls, df: pd.DataFrame, gold_model: str, is_skip_preprocess_if_exists: bool = False) -> pd.DataFrame:
        # there's nothing to add for the naive evaluator
        return df


class Checklister(base.BaseAnnotatorJSON):
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = object
    # the following line means that you can use both the rubric_eval or alpaca_eval models
    DEFAULT_BASE_DIR = CONFIGS_DIR / "checklisters_configs"
    annotator_column = "checklister"

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = ("instruction", "useful_info_to_eval_instruction"),
        annotators_config="gpt-4o-2024-08-06_v1",
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
        return "checklist"

    @property
    def dflt_annotator_kwargs(self):
        return dict(available_fields_to_format=self.available_fields_to_format, is_log_first_prompt=True)


class ChecklistSolutionEvaluator(BaseEvaluator):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto evaluator of the output using a gold checklist.",
    )
    DEFAULT_BASE_DIR = CONFIGS_DIR / "checklist_evaluators_configs"
    PRIMARY_KEYS = ("instruction", "output", "checklist", "solution")
    PREPROCESSING_COLUMNS = ["checklist", "solution"]
    PREPROCESSOR_COLUMNS = ["checklister", "solutioner"]

    @classmethod
    def preprocess(
        cls, df: pd.DataFrame, gold_model: str, is_skip_preprocess_if_exists: bool = False, **kwargs
    ) -> pd.DataFrame:
        if not cls._is_skip_preprocess(df, "solution", is_skip_preprocess_if_exists):
            df = add_solution_to_df(df, gold_model.split("_")[0], **kwargs)
        if cls._is_skip_preprocess(df, "checklist", is_skip_preprocess_if_exists):
            return df
        df["useful_info_to_eval_instruction"] = "### Good solution to the assignment:\n" + df["solution"]
        checklister = Checklister(annotators_config=gold_model.replace("_CoT", ""), **kwargs)
        checklists = checklister(df)
        df = ae_utils.convert_to_dataframe(checklists)
        return df


class ChecklistEvaluator(ChecklistSolutionEvaluator):
    @classmethod
    def preprocess(cls, df: pd.DataFrame, gold_model: str, is_skip_preprocess_if_exists: bool = False) -> pd.DataFrame:
        df = super().preprocess(df, gold_model, is_skip_preprocess_if_exists)
        df["solution"] = ""
        return df


class SolutionEvaluator(BaseEvaluator):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto evaluator of the output using the gold solution.",
    )
    DEFAULT_BASE_DIR = CONFIGS_DIR / "solution_evaluators_configs"
    PRIMARY_KEYS = (
        "instruction",
        "output",
        "solution",
    )
    PREPROCESSING_COLUMNS = ["solution"]
    PREPROCESSOR_COLUMNS = ["solutioner"]

    @classmethod
    def preprocess(
        cls, df: pd.DataFrame, gold_model: str, is_skip_preprocess_if_exists: bool = False, **kwargs
    ) -> pd.DataFrame:
        if cls._is_skip_preprocess(df, "solution", is_skip_preprocess_if_exists):
            return df
        df = add_solution_to_df(df, gold_model.split("_")[0], **kwargs)
        return df


class RubricSolutionEvaluator(REEvaluator):
    PREPROCESSING_COLUMNS = ["solution", "rubric"]
    PREPROCESSOR_COLUMNS = ["solutioner", "rubricator"]

    @classmethod
    def preprocess(
        cls, df: pd.DataFrame, gold_model: str, is_skip_preprocess_if_exists: bool = False, **kwargs
    ) -> pd.DataFrame:
        if not cls._is_skip_preprocess(df, "solution", is_skip_preprocess_if_exists):
            df = add_solution_to_df(df, gold_model.split("_")[0], **kwargs)
        if cls._is_skip_preprocess(df, "rubric", is_skip_preprocess_if_exists):
            df = Rubricator().make_df_rubrics(df, is_extract_criteria_col=True, is_renormalize_weight=True)
            return df
        df = df.drop(columns=["rubric"])
        df["useful_info_to_eval_instruction"] = "### Good solution to the assignment:\n" + df["solution"]
        df_with_brainstorm = main.brainstorm_rubrics_from_df(df, gold_model, **kwargs)
        df = main.generate_rubrics_from_df(df_with_brainstorm, gold_model, **kwargs)
        return df

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().make_df_rubric_grading(df)
        df["evaluator_type"] = self.__class__.__name__
        df["evaluator_seed"] = self.seed
        df["final_score"] = df["weighted_score"]
        return df

    @classmethod
    def _is_skip_preprocess(cls, df: pd.DataFrame, annotation_key: str, is_skip_preprocess_if_exists: bool) -> bool:
        is_skip = annotation_key in df.columns and df[annotation_key].notna().all() and is_skip_preprocess_if_exists
        if is_skip:
            logging.info(f"Skipping preprocess for {cls.__name__} because {annotation_key} is already present")
        return is_skip


class RubricEvaluator(RubricSolutionEvaluator):
    @classmethod
    def preprocess(cls, df: pd.DataFrame, gold_model: str, **kwargs) -> pd.DataFrame:
        df = super().preprocess(df, gold_model, **kwargs)
        # remove the response
        df["solution"] = ""
        return df


class ListRubricator(base.BaseAnnotatorJSON):
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = object
    # the following line means that you can use both the rubric_eval or alpaca_eval models
    DEFAULT_BASE_DIR = CONFIGS_DIR / "list_rubricators_configs"
    annotator_column = "list_rubricator"

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
            packages_for_which_to_show_version=["rubric_eval", "alpaca_eval"],
            **kwargs,
        )

    @property
    def annotation_key(self) -> str:
        return "list_error_rubric"

    @property
    def dflt_annotator_kwargs(self):
        return dict(available_fields_to_format=self.available_fields_to_format, is_log_first_prompt=True)


class ListRubricSolutionEvaluator(BaseEvaluator):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto evaluator of the output using a list-view of rubrics.",
    )
    DEFAULT_BASE_DIR = CONFIGS_DIR / "list_rubric_evaluators_configs"
    DEFAULT_ANNOTATION_TYPE = object
    PRIMARY_KEYS = (
        "instruction",
        "output",
        "list_error_rubric",
        "solution",
    )
    PREPROCESSING_COLUMNS = ["list_error_rubric", "solution"]
    PREPROCESSOR_COLUMNS = ["list_rubricator", "solutioner"]

    @classmethod
    def preprocess(
        cls, df: pd.DataFrame, gold_model: str, is_skip_preprocess_if_exists: bool = False, **kwargs
    ) -> pd.DataFrame:
        if not cls._is_skip_preprocess(df, "solution", is_skip_preprocess_if_exists):
            df = add_solution_to_df(df, gold_model.split("_")[0], **kwargs)
        if cls._is_skip_preprocess(df, "list_error_rubric", is_skip_preprocess_if_exists):
            return df
        df = df.drop(columns=["list_error_rubric"], errors="ignore")
        df["useful_info_to_eval_instruction"] = "### Good solution to the assignment:\n" + df["solution"]
        list_rubricator = ListRubricator(annotators_config=gold_model, **kwargs)
        list_rubrics = list_rubricator(df)
        df = ae_utils.convert_to_dataframe(list_rubrics)
        # df = helpers.expand_json_column(df, list_rubricator.annotation_key)
        return df

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df, is_expand_json_column=False)
        # "evaluation" is a list of dicts with keys "key", "description", "deductive_score"
        # now sum up the "deductive_score" for each key
        lower_bound = 1
        upper_bound = 10
        df["final_score"] = upper_bound + df["evaluation"].apply(lambda d: sum([v["delta_score"] for v in d]))
        df["final_score"] = df["final_score"].apply(lambda x: max(lower_bound, min(upper_bound, x)))
        return df


class ListRubricEvaluator(ListRubricSolutionEvaluator):
    @classmethod
    def preprocess(cls, df: pd.DataFrame, gold_model: str, is_skip_preprocess_if_exists: bool = False) -> pd.DataFrame:
        df = super().preprocess(df, gold_model, is_skip_preprocess_if_exists)
        # remove the response
        df["solution"] = ""
        return df


# HELPERS


def add_solution_to_df(df: pd.DataFrame, annotators_config: str, **ann_kwargs) -> pd.DataFrame:
    df = df.drop(columns=["solution"], errors="ignore")
    df = df.rename(columns={c: c.replace("output", "tokeep") for c in df.columns if c.startswith("output")})
    df["modelkeep"] = df["model"]
    # dirty trick to get the model name
    outputer = Outputer(annotators_config=annotators_config, **ann_kwargs)
    outputs = outputer(df)
    df = ae_utils.convert_to_dataframe(outputs)
    df["solutioner"] = df["model"]
    df["model"] = df["modelkeep"]
    # replace the new "output*" column with the "solution*"
    df = df.rename(columns={c: c.replace("output", "solution") for c in df.columns if c.startswith("output")})
    # replace the "tokeep*" columns with the "output*"
    df = df.rename(columns={c: c.replace("tokeep", "output") for c in df.columns if c.startswith("tokeep")})
    return df
