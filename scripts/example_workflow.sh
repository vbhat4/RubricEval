# bash scripts/example_workflow.sh

rm *.json
python scripts/convert_dataset_to_rubriceval_input_format.py  # This will generate an example instructions.json
rubric_eval get_rubrics --input_path=instructions.json
rubric_eval get_completions --model_config=gpt-4o-2024-05-13 --input_path=instructions_with_rubrics.json
rubric_eval --model_config=gpt-4o-2024-05-13 --input_path=completions.json