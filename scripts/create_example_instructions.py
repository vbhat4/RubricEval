from rubric_eval.helper import get_instructions

n_max_examples = 2
category = "Planning"
instruction_set = "wildbench-v1"
df_instructions = get_instructions(
    n_max_examples,
    category=category,
    instruction_set=instruction_set,
    with_additional_info=True,
)

# Save dataset to JSON file
df_instructions.to_json(f"instructions.json", orient='records', indent=4)