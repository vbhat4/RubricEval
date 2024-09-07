"""
File containing the 3 evaluators that are not rubric based:
- NaiveEvaluator # no conditioning
- ChecklistEvaluator # conditioning on a checklist of the rubric
- SolutionEvaluator # conditioning on the gold solution
"""
from pathlib import Path
from typing import Sequence

import pandas as pd
from alpaca_eval import utils as ae_utils
from alpaca_eval.annotators import base

from rubric_eval import helpers, main
from rubric_eval.evaluations import Evaluator as REEvaluator
from rubric_eval.outputs import Outputer

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
    DEFAULT_ANNOTATION_TYPE = "str"  # use object to have dict output
    annotator_column = "evaluator"
    DEFAULT_BASE_DIR = ...
    PRIMARY_KEYS = ...

    def __init__(
        self,
        *args,
        annotators_config: str = "gpt-4o-2024-08-06_CoT_v0",
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
    def preprocess(cls, df: pd.DataFrame, gold_model: str) -> pd.DataFrame:
        raise NotImplementedError


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

    @classmethod
    def preprocess(cls, df: pd.DataFrame, gold_model: str) -> pd.DataFrame:
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
        primary_keys: Sequence[str] = ("instruction",),
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
        return "checklist"


class ChecklistEvaluator(BaseEvaluator):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto evaluator of the output using a gold checklist.",
    )
    DEFAULT_BASE_DIR = CONFIGS_DIR / "checklist_evaluators_configs"
    PRIMARY_KEYS = (
        "instruction",
        "output",
        "checklist",
    )

    @classmethod
    def preprocess(cls, df: pd.DataFrame, gold_model: str) -> pd.DataFrame:
        checklister = Checklister(annotators_config=gold_model.replace("_CoT", ""))
        checklists = checklister(df)
        df = ae_utils.convert_to_dataframe(checklists)
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

    @classmethod
    def preprocess(cls, df: pd.DataFrame, gold_model: str) -> pd.DataFrame:
        # replace all the columns with "output*" to "tmp*" so that it's not lost
        df = df.rename(columns={c: c.replace("output", "tokeep") for c in df.columns if c.startswith("output")})
        df["modelkeep"] = df["model"]
        # dirty trick to get the model name
        outputer = Outputer(annotators_config=gold_model.split("_")[0])
        outputs = outputer(df)
        df = ae_utils.convert_to_dataframe(outputs)
        df["solutioner"] = df["model"]
        df["model"] = df["modelkeep"]
        # replace the new "output*" column with the "solution*"
        df = df.rename(columns={c: c.replace("output", "solution") for c in df.columns if c.startswith("output")})
        # replace the "tokeep*" columns with the "output*"
        df = df.rename(columns={c: c.replace("tokeep", "output") for c in df.columns if c.startswith("tokeep")})
        return df


class RubricSolutionEvaluator(REEvaluator):
    EXCELLENT_OUT_COL = "excellent_response"

    @classmethod
    def preprocess(cls, df: pd.DataFrame, gold_model: str, **kwargs) -> pd.DataFrame:
        df = df.drop(columns=["rubric", cls.EXCELLENT_OUT_COL])
        df_with_brainstorm = main.brainstorm_rubrics_from_df(df, gold_model, **kwargs)
        df = main.generate_rubrics_from_df(df_with_brainstorm, gold_model, **kwargs)
        if cls.EXCELLENT_OUT_COL in df.columns:
            # some models do not return the excellent response, we fill it with an empty string
            df[cls.EXCELLENT_OUT_COL] = df[cls.EXCELLENT_OUT_COL].fillna("")
        return df

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().make_df_rubric_grading(df)
        df["evaluator_type"] = self.__class__.__name__
        df["evaluator_seed"] = self.seed
        df["final_score"] = df["weighted_score"]
        return df


class RubricEvaluator(RubricSolutionEvaluator):
    @classmethod
    def preprocess(cls, df: pd.DataFrame, gold_model: str, **kwargs) -> pd.DataFrame:
        df = super().preprocess(df, gold_model, **kwargs)
        # remove the response
        df[cls.EXCELLENT_OUT_COL] = ""
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
    def annotation_key(self) -> str:
        return "list_error_rubric"


class ListRubricSolutionEvaluator(BaseEvaluator):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto evaluator of the output using a list-view of rubrics.",
    )
    DEFAULT_BASE_DIR = CONFIGS_DIR / "list_rubric_evaluators_configs"
    DEFAULT_ANNOTATION_TYPE = object
    EXCELLENT_OUT_COL = "strong_response"
    PRIMARY_KEYS = (
        "instruction",
        "output",
        "list_error_rubric",
        EXCELLENT_OUT_COL,
    )

    @classmethod
    def preprocess(cls, df: pd.DataFrame, gold_model: str) -> pd.DataFrame:
        df = df.drop(columns=["list_error_rubric", cls.EXCELLENT_OUT_COL], errors="ignore")
        list_rubricator = ListRubricator(annotators_config=gold_model)
        list_rubrics = list_rubricator(df)
        df = ae_utils.convert_to_dataframe(list_rubrics)
        df = helpers.expand_json_column(df, list_rubricator.annotation_key)
        if cls.EXCELLENT_OUT_COL in df.columns:
            # some models do not return the excellent response, we fill it with an empty string
            df[cls.EXCELLENT_OUT_COL] = df[cls.EXCELLENT_OUT_COL].fillna("")
        return df

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df, is_expand_json_column=False)
        # "evaluation" is a list of dicts with keys "key", "description", "deductive_score"
        # now sum up the "deductive_score" for each key
        lower_bound = 1
        upper_bound = 10
        try:
            df["final_score"] = upper_bound + df["evaluation"].apply(lambda d: sum([v["delta_score"] for v in d]))
        except Exception as e:
            breakpoint()
        df["final_score"] = df["final_score"].apply(lambda x: max(lower_bound, min(upper_bound, x)))
        return df


class ListRubricEvaluator(ListRubricSolutionEvaluator):
    @classmethod
    def preprocess(cls, df: pd.DataFrame, gold_model: str) -> pd.DataFrame:
        df = super().preprocess(df, gold_model)
        # remove the response
        df[cls.EXCELLENT_OUT_COL] = ""
        return df
