import pandas as pd
import tempfile, os, json
from pathlib import Path
import unittest
from unittest.mock import patch
from rubric_eval.main import get_rubrics, get_completions, evaluate

class TestMain(unittest.TestCase):
    def setUp(self):
        self.test_data_path = Path(__file__).resolve().parent.parent / 'tests' / 'test_data'

    @patch('alpaca_eval.decoders.openai._openai_completion_helper')
    def test_get_rubrics_df_input(self, mock_completion_helper):
        # Test with DataFrame input and output
        completion_mock = dict()
        completion_mock["total_tokens"] = 3
        completion_mock["text"] = '''{"scoring_scales":{"Excellent":4,"Good":3,"Fair":2,"Poor":1},"detailed_analytic_rubric":{"Narrative Structure":{"Excellent":"The story exhibits a clear and compelling beginning, middle, and end. The plot is coherent, logically structured, and free of plot holes or unresolved storylines. The transition between the story\'s phases is smooth, ensuring a seamless narrative flow.","Good":"The story has a recognizable beginning, middle, and end, but may lack smooth transitions. The plot is mostly coherent with minor logical inconsistencies or minor unresolved storylines.","Fair":"The story\'s structure is present but underdeveloped, with a vague beginning, middle, or end. The plot shows noticeable logical inconsistencies, and there may be significant unresolved storylines.","Poor":"The story lacks a discernible beginning, middle, and end. The plot is incoherent, illogically structured, and filled with unresolved storylines or significant plot holes."},"Character Development":{"Excellent":"The cat\'s personality is vividly and consistently portrayed throughout the story, showcasing significant development or change that is believable and well-motivated. The cat\'s actions and motivations are consistent, contributing to a deep understanding of its character.","Good":"The cat\'s personality is clear, and there is some development or change by the end of the story. However, the development may be somewhat predictable or not fully explored. The cat\'s actions and motivations are generally consistent.","Fair":"The cat\'s personality is somewhat defined, but the development or change is minimal or unconvincing. The cat\'s actions and motivations lack consistency, making the character feel less believable.","Poor":"The cat\'s personality is poorly defined, with no discernible development or change. The cat\'s actions and motivations are inconsistent or illogical, failing to contribute to a coherent character portrayal."},"Creativity and Originality":{"Excellent":"The story is highly original and creative, featuring a unique and engaging plot with unexpected twists or magical elements that add depth. It is unpredictable and surprises the reader, standing out for its originality.","Good":"The story has some original elements and creative aspects, but may rely on familiar tropes or predictable plot developments. It offers a degree of surprise or creativity that enhances the narrative.","Fair":"The story shows limited creativity, with a plot that feels derivative or predictable. There are few if any, original elements or creative twists, making the story feel somewhat stale.","Poor":"The story lacks originality, with a completely predictable plot that does not deviate from well-worn tropes. There is no evidence of creativity or effort to introduce unique elements or twists."},"Descriptive Language and Setting":{"Excellent":"The settings are vividly described, significantly enhancing the reader\'s immersion. The language effectively conveys the mood and tone of the story, with sensory details that bring scenes to life.","Good":"The settings are described well enough to aid in immersion, and the language conveys the mood and tone, though with less impact. Sensory details are used, but not to their fullest potential to enhance the narrative.","Fair":"The descriptions of settings are basic, providing minimal enhancement to the reader\'s immersion. The language conveys some aspects of the mood and tone, but lacks depth. Sensory details are sparse or ineffectively used.","Poor":"The settings are poorly described or not described at all, failing to contribute to the reader\'s immersion. The language does not effectively convey the mood or tone, and there is a lack of sensory details, making scenes feel flat and lifeless."},"General Quality":{"Excellent":"The response is clear, concise, well-organized, and maintains the reader\'s interest throughout. The writing is free of grammatical and spelling errors and is safe and appropriate for all audiences.","Good":"The response is mostly clear and organized, with minor issues in clarity or organization that slightly detract from the reader\'s interest. There are few grammatical or spelling errors, and the content is generally safe and appropriate.","Fair":"The response is somewhat clear and organized, but there are noticeable issues in clarity, organization, or interest maintenance. There are several grammatical or spelling errors, and the content may not be entirely appropriate for all audiences.","Poor":"The response is unclear, disorganized, and fails to maintain the reader\'s interest. There are numerous grammatical and spelling errors, and the content may be inappropriate for some audiences."}}}'''
        mock_completion_helper.return_value = [completion_mock]
        
        input_df = pd.DataFrame({'prompt': ['Write a short story about a cat.']})
        output_df = get_rubrics(input_df)
        self.assertTrue(isinstance(output_df, pd.DataFrame))
        self.assertTrue('scoring_scales' in output_df.columns)
        
        mock_completion_helper.assert_called_once()
    
    @patch('alpaca_eval.decoders.openai._openai_completion_helper')
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

    @patch('alpaca_eval.decoders.openai._openai_completion_helper')
    def test_get_completions_df_input(self):
        # Test with DataFrame input and output
        input_path = self.test_data_path / 'instructions_with_rubrics.json'
        input_df = pd.read_json(input_path)
        output_df = get_completions("gpt-4o-2024-05-13", input_df)
        self.assertTrue(isinstance(output_df, pd.DataFrame))
        self.assertTrue('output' in output_df.columns)

    @patch('alpaca_eval.decoders.openai._openai_completion_helper')
    def test_get_completions_file_input(self):
        # Test with file input and output
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = self.test_data_path / 'instructions_with_rubrics.json'
            output_path = Path(tmpdir) / 'output.json'
            get_completions("gpt-4o-2024-05-13", input_path=input_path, output_path=output_path)
            output_df = pd.read_json(output_path)
            self.assertTrue('output' in output_df.columns)

    @patch('alpaca_eval.decoders.openai._openai_completion_helper')
    def test_evaluate_df_input(self):
        # Test with DataFrame input and output
        input_path = self.test_data_path / 'completions.json'
        input_df = pd.read_json(input_path)
        output_df = evaluate("gpt-4o-2024-05-13", input_df)
        self.assertTrue(isinstance(output_df, pd.DataFrame))
        self.assertTrue('criteria_scores' in output_df.columns)

    @patch('alpaca_eval.decoders.openai._openai_completion_helper')
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