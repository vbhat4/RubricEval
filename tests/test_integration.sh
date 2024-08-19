#!/bin/bash
set -e

if [ -z "${OPENAI_API_KEY+x}" ]; then
    OPENAI_API_KEY="your_api_key_here"  # Replace "your_api_key_here" with your OpenAI API key in order to run this script
fi

# Pre-flight checks
python -c "from tests.test_integration import check_evaluate_outputs"

# Test file I/O
rm -rf tmp_dir && mkdir tmp_dir/
python scripts/create_example_instructions.py tmp_dir/instructions.json  # This will generate an example instructions.json
rubric_eval get_rubrics --input_path=tmp_dir/instructions.json --output_path=tmp_dir/instructions_with_rubrics.json --cache_dir=None
rubric_eval generate_outputs --model_configs=gpt-4o-2024-05-13 --input_path=tmp_dir/instructions_with_rubrics.json --output_path=tmp_dir/completions.json --cache_dir=None
rubric_eval --model_configs=gpt-4o-2024-05-13 --input_path=tmp_dir/completions.json --output_path=tmp_dir/evaluations.json --cache_dir=None
python -c "from tests.test_integration import check_evaluate_outputs; check_evaluate_outputs('tmp_dir/evaluations.json', 'tmp_dir/model_card.json')"
rm -rf tmp_dir/

# Test Python function calling
python tests/test_integration.py

# Run all doctests
python src/rubric_eval/main.py run_doctests

# Run all unit tests
python tests/test_main.py
python tests/test_helper.py

echo "Integration tests passed!"
