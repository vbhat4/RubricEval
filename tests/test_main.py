import pandas as pd
import tempfile, os, json
from pathlib import Path
import unittest
from rubric_eval.main import get_rubrics, get_completions, evaluate

class TestMain(unittest.TestCase):
    def setUp(self):
        self.test_data_path = Path(__file__).resolve().parent.parent / 'tests' / 'test_data'

    def test_get_rubrics_df_input(self):
        # Test with DataFrame input and output
        input_df = pd.DataFrame({'prompt': ['Write a short story about a cat.']})
        output_df = get_rubrics(input_df)
        self.assertTrue(isinstance(output_df, pd.DataFrame))
        self.assertTrue('scoring_scales' in output_df.columns)
    
    def test_get_rubrics_file_input(self):
        # Test with file input and output
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'input.json'
            with open(input_path, 'w') as f:
                json.dump([{'prompt': 'Write a short story about a cat.'}], f)
            output_path = Path(tmpdir) / 'output.json'
            get_rubrics(input_path=input_path, output_path=output_path)
            output_df = pd.read_json(output_path)
            self.assertTrue('scoring_scales' in output_df.columns)

    def test_get_completions_df_input(self):
        # Test with DataFrame input and output
        input_path = self.test_data_path / 'instructions_with_rubrics.json'
        input_df = pd.read_json(input_path)
        output_df = get_completions("gpt-4o-2024-05-13", input_df)
        self.assertTrue(isinstance(output_df, pd.DataFrame))
        self.assertTrue('output' in output_df.columns)

    def test_get_completions_file_input(self):
        # Test with file input and output
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = self.test_data_path / 'instructions_with_rubrics.json'
            output_path = Path(tmpdir) / 'output.json'
            get_completions("gpt-4o-2024-05-13", input_path=input_path, output_path=output_path)
            output_df = pd.read_json(output_path)
            self.assertTrue('output' in output_df.columns)

    def test_evaluate_df_input(self):
        # Test with DataFrame input and output
        input_path = self.test_data_path / 'completions.json'
        input_df = pd.read_json(input_path)
        output_df = evaluate("gpt-4o-2024-05-13", input_df)
        self.assertTrue(isinstance(output_df, pd.DataFrame))
        self.assertTrue('criteria_scores' in output_df.columns)

    def test_evaluate_file_input(self):
        # Test with file input and output
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = self.test_data_path / 'completions.json'
            output_path = Path(tmpdir) / 'output.json'
            evaluate("gpt-4o-2024-05-13", input_path=input_path, output_path=output_path)
            output_df = pd.read_json(output_path)
            self.assertTrue('criteria_scores' in output_df.columns)

if __name__ == '__main__':
    unittest.main()