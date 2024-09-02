import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from rubric_eval.helper import get_instructions
from rubric_eval.main import evaluate, generate_outputs, generate_rubrics


def check_evaluate_outputs(evaluations, model_card):
    if isinstance(evaluations, pd.DataFrame):
        assert isinstance(model_card, pd.DataFrame)
        df_evaluations = evaluations
        df_model_card = model_card
    else:
        df_evaluations = pd.read_json(evaluations)
        df_model_card = pd.read_json(model_card)
    assert isinstance(df_evaluations, pd.DataFrame)
    assert isinstance(df_model_card, pd.DataFrame)
    assert "criteria_scores" in df_evaluations.columns
    assert "mean_of_avg_score" in df_model_card.columns
    assert df_model_card["mean_of_avg_score"].iloc[0] < 4.0
    assert "std_of_avg_score" in df_model_card.columns
    assert df_model_card["std_of_avg_score"].iloc[0] > 0.0


class TestIntegration(unittest.TestCase):
    def test_integration(self):
        n_max_examples = 5
        category = "Planning"
        instruction_set = "wildbench-v1"
        df_instructions = get_instructions(
            n_max_examples,
            category=category,
            instruction_set=instruction_set,
            with_additional_info=True,
            random_seed=123,
        )
        self.assertTrue("prompt" in df_instructions.columns)

        df_rubrics = generate_rubrics(df_instructions)
        self.assertTrue("scoring_scales" in df_rubrics.columns)

        df_outputs = generate_outputs("gpt-4o-2024-05-13", df_rubrics)
        self.assertTrue("output" in df_outputs.columns)

        df_evaluations, df_model_card = evaluate("gpt-4o-2024-05-13", df_outputs)
        check_evaluate_outputs(df_evaluations, df_model_card)


if __name__ == "__main__":
    unittest.main()
