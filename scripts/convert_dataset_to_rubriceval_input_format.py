from rubric_eval.helper import get_instructions
from alpaca_eval import utils as ae_utils

n_max_examples = 2
category = "Planning"
instruction_set = "wildbench"
instructions = get_instructions(
    n_max_examples,
    category=category,
    instruction_set=instruction_set,
    with_additional_info=True,
)
df = ae_utils.convert_to_dataframe(instructions)

# Save dataset to JSON file
df.to_json(f"instructions.json", orient='records', indent=4)