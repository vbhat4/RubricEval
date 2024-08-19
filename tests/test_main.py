import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from rubric_eval.main import evaluate, generate_outputs, generate_rubrics


class TestMain(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = Path(__file__).resolve().parent / "test_data"

    def test_generate_rubrics_df_input(self):
        # Test with DataFrame input and output
        instructions_path = self.test_data_dir / "instructions.json"
        df_instructions = pd.read_json(instructions_path)
        df_rubrics = generate_rubrics(df_instructions, cache_dir=self.test_data_dir)
        self.assertTrue(isinstance(df_rubrics, pd.DataFrame))
        self.assertTrue("scoring_scales" in df_rubrics.columns)

    def test_generate_rubrics_file_input(self):
        # Test with file input and output
        with tempfile.TemporaryDirectory() as tmpdir:
            instructions_path = self.test_data_dir / "instructions.json"
            instructions_with_rubrics_path = Path(tmpdir) / "instructions_with_rubrics.json"
            generate_rubrics(
                input_path=instructions_path, output_path=instructions_with_rubrics_path, cache_dir=self.test_data_dir
            )
            output_df = pd.read_json(instructions_with_rubrics_path)
            self.assertTrue("scoring_scales" in output_df.columns)

    def test_generate_outputs_df_input(self):
        # Test with DataFrame input and output
        instructions_with_rubrics_path = self.test_data_dir / "instructions_with_rubrics.json"
        df_instruction_with_rubrics = pd.read_json(instructions_with_rubrics_path)
        df_outputs = generate_outputs("gpt-4o-2024-05-13", df_instruction_with_rubrics, cache_dir=self.test_data_dir)
        self.assertTrue(isinstance(df_outputs, pd.DataFrame))
        self.assertTrue("output" in df_outputs.columns)

    def test_generate_outputs_file_input(self):
        # Test with file input and output
        with tempfile.TemporaryDirectory() as tmpdir:
            instructions_with_rubrics_path = self.test_data_dir / "instructions_with_rubrics.json"
            completions_path = Path(tmpdir) / "completions.json"
            generate_outputs(
                "gpt-4o-2024-05-13",
                input_path=instructions_with_rubrics_path,
                output_path=completions_path,
                cache_dir=self.test_data_dir,
            )
            df_outputs = pd.read_json(completions_path)
            self.assertTrue("output" in df_outputs.columns)

    def test_evaluate_df_input(self):
        # Test with DataFrame input and output
        completions_path = self.test_data_dir / "completions.json"
        df_outputs = pd.read_json(completions_path)
        df_evaluations, df_model_card = evaluate("gpt-4o-2024-05-13", df_outputs, cache_dir=self.test_data_dir)
        self.assertTrue(isinstance(df_evaluations, pd.DataFrame))
        self.assertTrue(isinstance(df_model_card, pd.DataFrame))
        self.assertTrue("criteria_scores" in df_evaluations.columns)
        self.assertTrue("mean_of_avg_score" in df_model_card.columns)
        self.assertTrue(df_model_card["mean_of_avg_score"].iloc[0] < 4.0)
        self.assertTrue("std_of_avg_score" in df_model_card.columns)
        self.assertTrue(df_model_card["std_of_avg_score"].iloc[0] > 0.0)

    def test_evaluate_file_input(self):
        # Test with file input and output
        with tempfile.TemporaryDirectory() as tmpdir:
            completions_path = self.test_data_dir / "completions.json"
            evaluations_path = Path(tmpdir) / "evaluations.json"
            evaluate(
                "gpt-4o-2024-05-13",
                input_path=completions_path,
                output_path=evaluations_path,
                cache_dir=self.test_data_dir,
            )
            df_evaluations = pd.read_json(evaluations_path)
            model_card_path = Path(tmpdir) / "model_card.json"
            df_model_card = pd.read_json(model_card_path)
            self.assertTrue(isinstance(df_evaluations, pd.DataFrame))
            self.assertTrue(isinstance(df_model_card, pd.DataFrame))
            self.assertTrue("criteria_scores" in df_evaluations.columns)
            self.assertTrue("mean_of_avg_score" in df_model_card.columns)
            self.assertTrue(df_model_card["mean_of_avg_score"].iloc[0] < 4.0)
            self.assertTrue("std_of_avg_score" in df_model_card.columns)
            self.assertTrue(df_model_card["std_of_avg_score"].iloc[0] > 0.0)


if __name__ == "__main__":
    unittest.main()
