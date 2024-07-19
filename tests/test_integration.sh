#!/bin/bash

# Test file I/O
export OPENAI_API_KEY=<your_api_key>
rm -rf tmp_dir && mkdir tmp_dir/
python scripts/create_example_instructions.py tmp_dir/instructions.json  # This will generate an example instructions.json
rubric_eval get_rubrics --input_path=tmp_dir/instructions.json --output_path=tmp_dir/instructions_with_rubrics.json --cache_dir=None
rubric_eval get_completions --model_config=gpt-4o-2024-05-13 --input_path=tmp_dir/instructions_with_rubrics.json --output_path=tmp_dir/completions.json --cache_dir=None
rubric_eval --model_config=gpt-4o-2024-05-13 --input_path=tmp_dir/completions.json --output_path=tmp_dir/evaluations.json --cache_dir=None
python3 << END
import pandas as pd
from pathlib import Path

evaluations_path = Path('tmpdir') / 'evaluations.json'
evaluations_df = pd.read_json(evaluations_path)
self.assertTrue(isinstance(evaluations_df, pd.DataFrame))
self.assertTrue('criteria_scores' in evaluations_df.columns)

model_card_path = Path('tmpdir') / 'model_card.json'
model_card_df = pd.read_json(model_card_path)
self.assertTrue(isinstance(model_card_df, pd.DataFrame))
self.assertTrue('mean_of_avg_score' in model_card_df.columns)
self.assertTrue(model_card_df['mean_of_avg_score'].iloc[0] < 4.0)
self.assertTrue('std_of_avg_score' in model_card_df.columns)
self.assertTrue(model_card_df['std_of_avg_score'].iloc[0] > 0.0)
END


# Test Python function calling
python3 ./test_integration.py

# Run all doctests
python src/rubric_eval/main.py run_doctests
