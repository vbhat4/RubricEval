import pandas as pd
import tempfile, os, json
from pathlib import Path
import unittest
from unittest.mock import patch
from rubric_eval.main import get_rubrics, get_completions, evaluate
from scripts.create_example_instructions import create_example_instructions


class TestIntegration(unittest.TestCase):
    def test_integration(self):
        df_instructions = create_example_instructions()
        
        df_rubrics = get_rubrics(df_instructions)
        self.assertTrue('scoring_scales' in df_rubrics.columns)
        
        df_completions = get_completions("gpt-4o-2024-05-13", df_rubrics)
        self.assertTrue('output' in df_completions.columns)
        
        df_evaluations, df_model_card = evaluate("gpt-4o-2024-05-13", df_completions)
        self.assertTrue('criteria_scores' in df_evaluations.columns)
        self.assertTrue('mean_of_avg_score' in df_model_card.columns)
        self.assertTrue(df_model_card['mean_of_avg_score'].iloc[0] < 4.0)
        self.assertTrue('std_of_avg_score' in df_model_card.columns)
        self.assertTrue(df_model_card['std_of_avg_score'].iloc[0] > 0.0)


if __name__ == '__main__':
    unittest.main()