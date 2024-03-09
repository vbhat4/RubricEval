from pathlib import Path
from typing import Sequence, Dict

from alpaca_eval import utils as ae_utils
import ast
import pandas as pd

from alpaca_eval.annotators import base

__all__ = ["Instructionator", "Rubricator", "Completor", "Evaluator"]


class Instructionator(base.BaseAnnotatorJSON):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto annotator for writing the initial instruction.",
    )
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = str
    DEFAULT_BASE_DIR = Path(__file__).parent / "instructionator_configs"

    CATEGORIES: Dict[str, str] = {
        "Creativity": "This involves generating novel, useful, and unexpected ideas or concepts. This can be assessed through tasks such as story generation, poem writing, creating new game rules, or developing innovative product ideas.",
        "Math & Logical Reasoning": "This involves the capacity to understand, analyze, and solve problems using mathematical concepts, formal logic, and deductive reasoning.",
        "Coding": "Check the ability to understand, generate, and debug code in various programming languages.",
        "Factual Knowledge": "This relates to the accurate recall and understanding of concrete facts and information, often about the world, history, science, culture, etc.",
        "Common Sense Reasoning": "This involves applying basic, intuitive understanding of everyday situations and events, often involving implicit knowledge that humans typically take for granted.",
        "Task Completion": "Evaluate the model's ability to carry out specific tasks, such as summarizing a text or changing the format of a document.",
        "Adaptability to Different Domains & Role Playing": "Assess how well the model can understand and generate text related to specific domains, such as medicine, law, finance, etc.",
        "Ethical Reasoning": "This refers to the capacity to consider, evaluate, and make decisions based on moral principles and ethical guidelines.",
        "Emotional Intelligence": "This involves understanding, interpreting, and responding to emotions in oneself and others, often linked to empathy, self-awareness, and social skills.",
        "Multi-language Proficiency": "Evaluate how well the model performs across different languages, not just in terms of fluency, but also in understanding cultural nuances, idioms, and colloquialisms.",
        "Robustness": "Test the model's ability to understand and respond appropriately even in the presence of noise(e.g. typos, grammatical errors, ...), conflicting pieces of information, ambiguity, missing or irrelevant information, and other adversarial inputs.",
    }

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = ("n", "category", "categories"),
        annotators_config: str = "gpt4_CoT_v0",
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
        return "categories_and_instructions"

    def _generate_list(self) -> str:
        return "\n".join(
            [f" - {key}: {value}" for key, value in self.CATEGORIES.items()]
        )

    def generate_n_instruction_for_a_category(
        self, n: int, category: str
    ) -> Sequence[dict]:
        input = [{"n": n, "category": category, "categories": self._generate_list()}]
        return self(input)


class Rubricator(base.BaseAnnotatorJSON):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto annotator for writing the rubric.",
    )
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = str
    DEFAULT_BASE_DIR = Path(__file__).parent / "rubricator_configs"

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = ("prompt",),
        annotators_config="gpt4_CoT_v0",
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
        return "assignment_and_rubric"

    def make_df_rubrics(self, annotated: Sequence[dict]) -> pd.DataFrame:
        df_rubrics = ae_utils.convert_to_dataframe(annotated)
        df_rubrics = pd.concat(
            [
                df_rubrics.drop([self.annotation_key], axis=1),
                pd.json_normalize(
                    df_rubrics[self.annotation_key].apply(ast.literal_eval), max_level=0
                ),
            ],
            axis=1,
        )
        return df_rubrics


# TODO: should likely be in a different file and not inherit from BaseAnnotatorJSON.
class Completor(base.BaseAnnotatorJSON):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Model to evaluate.",
    )
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = str
    DEFAULT_BASE_DIR = Path(__file__).parent / "completor_configs"

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = ("final_prompt",),
        annotators_config="claude-2",
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
        return "output"


class Evaluator(base.BaseAnnotatorJSON):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto evaluator of the output using the rubric.",
    )
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = "object"  # use object to have dict output
    DEFAULT_BASE_DIR = Path(__file__).parent / "evaluator_configs"

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = (
            "final_prompt",
            "detailed_analytic_rubric",
            "output",
            "key_criteria",
            "scoring_scales",
        ),
        annotators_config="gpt4_CoT_v0",
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
        return "criteria_scores"
