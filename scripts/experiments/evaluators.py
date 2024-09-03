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

__all__ = ["NaiveEvaluator", "ChecklistEvaluator", "SolutionEvaluator", "RubricEvaluator", "BaseEvaluator"]

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

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = helpers.expand_json_column(df, self.annotation_key)
        df["evaluator_type"] = self.__class__.__name__
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


class _Checklister(base.BaseAnnotatorJSON):
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = str
    # the following line means that you can use both the rubric_eval or alpaca_eval models
    DEFAULT_BASE_DIR = CONFIGS_DIR / "checklister_configs"
    annotator_column = "checklister"

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
        checklister = _Checklister(annotators_config=gold_model.replace("_CoT", ""))
        checklists = checklister(df)
        df = ae_utils.convert_to_dataframe(checklists)
        df["checklist"] = df["checklist"].apply(lambda x: "\n".join(x))

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
        # dirty trick to get the model name
        outputer = Outputer(annotators_config=gold_model.split("_")[0])
        outputs = outputer(df)
        df = ae_utils.convert_to_dataframe(outputs)
        # replace the new "output*" column with the "solution*"
        df = df.rename(columns={c: c.replace("output", "solution") for c in df.columns if c.startswith("output")})
        # replace the "tokeep*" columns with the "output*"
        df = df.rename(columns={c: c.replace("tokeep", "output") for c in df.columns if c.startswith("tokeep")})
        return df


class RubricEvaluator(REEvaluator):
    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().make_df_rubric_grading(df)
        df["evaluator_type"] = self.__class__.__name__
        # df["weighted_score"]  is between between 1-4, we want between 1-10
        df["final_score"] = (df["weighted_score"] - 1) * 3 + 1
        return df

    @classmethod
    def preprocess(cls, df: pd.DataFrame, gold_model: str, **kwargs) -> pd.DataFrame:
        df_with_brainstorm = main.brainstorm_rubrics_from_df(df, gold_model, **kwargs)
        df_with_rubrics = main.generate_rubrics_from_df(df_with_brainstorm, gold_model, **kwargs)
        return df_with_rubrics
