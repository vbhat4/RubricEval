import sys
from typing import Optional
from rubric_eval.helper import get_instructions, get_instructions


if __name__ == "__main__":
    assert len(sys.argv) > 1 and len(sys.argv) <= 2, "Usage: python create_example_instructions.py [output_path]"
    output_path = sys.argv[1]
    n_max_examples = 5
    category = "Planning"
    instruction_set = "wildbench-v1"
    get_instructions(
        n_max_examples,
        category=category,
        instruction_set=instruction_set,
        with_additional_info=True,
        random_seed=123,
        output_path=output_path,
    )