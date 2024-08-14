from helm_instruct.helper import *

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Cell 3
    n_max_examples = 24
    instruction_set = "wildbench"

    # Cell 4
    instructions= get_instructions(n_max_examples,
                                                instruction_set=instruction_set,
                                                sample_by_category=True,
                                                with_additional_info=False,
                                                n_to_print=0)
    instructions_with_add_info = get_instructions(n_max_examples,
                                                instruction_set=instruction_set,
                                                sample_by_category=True,
                                                with_additional_info=True,
                                                n_to_print=0)
    # ambiguous and overly complex instructions
    filtered_indices = [3, 5, 6, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23]
    instructions = [instructions[i] for i in range(len(instructions)) if i not in filtered_indices]
    instructions_with_add_info = [instructions_with_add_info[i] for i in range(len(instructions_with_add_info)) if i not in filtered_indices]

    # Cell 7
    df_criteria = get_criteria(instructions[:5], n_to_print=0)

    # Cell 9
    df_criteria_with_add_info = get_criteria(instructions_with_add_info, n_to_print=0)

    # Cell 10
    df_rubrics = get_detailed_rubrics(df_criteria[3:], n_to_print=0, is_store_missing_annotations=True)

    # Cell 12
    df_rubrics.criteria = df_rubrics.criteria.apply(lambda x: [d['aspect'] for d in x])

    # Cell 16
    llama3_70b_completions = get_model_completions(df_rubrics, "Meta-Llama-3-70B-Instruct", n_to_print=2)
    print(f"llama3_70b_completions: {llama3_70b_completions}")