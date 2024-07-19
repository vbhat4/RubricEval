import sys
from typing import Optional
from rubric_eval.helper import get_instructions
from alpaca_eval.types import AnyPath


def create_example_instructions(output_path: Optional[AnyPath] = None):
    n_max_examples = 5
    category = "Planning"
    instruction_set = "wildbench-v1"
    df_instructions = get_instructions(
        n_max_examples,
        category=category,
        instruction_set=instruction_set,
        with_additional_info=True,
        random_seed=123,
    )
    if output_path is not None:
        df_instructions.to_json(output_path, orient='records', indent=4)
    else:
        return df_instructions


if __name__ == "__main__":
    assert len(sys.argv) > 1 and len(sys.argv) <= 2, "Usage: python create_example_instructions.py [output_path]"
    output_path = sys.argv[1]
    create_example_instructions(output_path)