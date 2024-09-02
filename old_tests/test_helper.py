import pandas as pd
import tempfile, os, json
from pathlib import Path
import unittest
from rubric_eval.helper import get_instructions

class TestHelper(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = Path(__file__).resolve().parent.parent / 'tests' / 'test_data'

    def test_get_instructions_auto(self):
        df = get_instructions(n_max_examples=2, instruction_set='auto', category='Planning', cache_dir=self.test_data_dir)
        self.assertEqual(len(df), 2)
        self.assertTrue('prompt' in df.columns)

    def test_get_instructions_alpaca_eval_2(self):
        df = get_instructions(n_max_examples=2, instruction_set='alpaca_eval_2', random_seed=123)
        self.assertEqual(len(df), 2)
        self.assertTrue('prompt' in df.columns)

    def test_get_instructions_wildbench_v1(self):
        df = get_instructions(n_max_examples=2, instruction_set='wildbench-v1', category="Planning", random_seed=123)
        self.assertEqual(len(df), 2)
        self.assertTrue('prompt' in df.columns)

        df = get_instructions(n_max_examples=2, instruction_set='wildbench-v1', category=None, random_seed=123)
        self.assertEqual(len(df), 2)
        self.assertTrue('prompt' in df.columns)


if __name__ == '__main__':
    unittest.main()