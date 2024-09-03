from typing import Sequence

from alpaca_eval import constants as ae_constants
from alpaca_eval.annotators import base

from .helpers import CONFIGS_DIR

__all__ = ["Outputer"]


class Outputer(base.BaseAnnotatorJSON):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Model to evaluate.",
    )
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = str
    # the following line means that you can use both the rubric_eval or alpaca_eval models
    DEFAULT_BASE_DIR = [CONFIGS_DIR / "models_configs", ae_constants.MODELS_CONFIG_DIR]
    annotator_column = "model"

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = ("instruction",),
        annotators_config="gpt-4o-2024-05-13",
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
        return "output"
