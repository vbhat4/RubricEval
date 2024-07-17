import pandas as pd
import tempfile, os, json
from pathlib import Path
import unittest
from rubric_eval.main import preprocess_df_instructions
from rubric_eval.helper import get_detailed_rubrics, get_model_completions
from rubric_eval.annotators import RubricBrainstormer
from alpaca_eval import utils as ae_utils

class TestHelper(unittest.TestCase):
    def setUp(self):
        self.test_data_path = Path(__file__).resolve().parent.parent / 'tests' / 'test_data'

    def test_get_instructions(self):
        # TODO
        pass
    
    def test_get_detailed_rubrics(self):
        df = pd.DataFrame({'prompt': ['Write a short story about a cat.']})
        df = preprocess_df_instructions(df)
        rubric_generator = "gpt4_CoT_v0"
        rubric_brainstormer = RubricBrainstormer(annotators_config=rubric_generator)
        criteria = rubric_brainstormer(df)
        df_criteria = rubric_brainstormer.make_df_rubrics(criteria)
        df_rubrics = get_detailed_rubrics(df_criteria, is_store_missing_annotations=True, annotators_config=rubric_generator)
        self.assertTrue(isinstance(df_rubrics, pd.DataFrame))
        self.assertTrue('scoring_scales' in df_rubrics.columns)

    def test_get_model_completions(self):
        df_rubrics = pd.read_json(self.test_data_path / 'instructions_with_rubrics.json')
        completions = get_model_completions(df_rubrics, "gpt-4o-2024-05-13")
        df_completions = ae_utils.convert_to_dataframe(completions)
        self.assertTrue(isinstance(df_completions, pd.DataFrame))
        self.assertTrue('output' in df_completions.columns)

    def test_get_evaluations(self):
        # TODO
        pass

if __name__ == '__main__':
    unittest.main()