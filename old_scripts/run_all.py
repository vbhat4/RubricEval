"""
clear && clear && time TOGETHER_API_KEY=<together_api_key> python scripts/run_all.py  >output1.txt 2>&1
"""

from rubric_eval.helper import *
import pickle
from pathlib import Path

model_names = [
    "gpt-4o-2024-05-13",
    "chatgpt",

    "gemini-1.5-pro-001",
    "gemini-1.5-flash-001",

    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    
    "Meta-Llama-3-70B-Instruct",
    "Meta-Llama-3-8B-Instruct",
    
    "Mixtral-8x22B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.3",
    "Qwen2-72B-Instruct",
]

evaluator_names = ["gpt-4o-2024-05-13_CoT_v0"]
categories = ["Advice seeking", "Reasoning", "Editing", "Planning", "Information seeking", "Creative Writing", "Coding & Debugging", "Brainstorming", "Math", "Role playing", "Data Analysis", "Others"]
exp_suffixes = ["rerun_exp_v5", "no_rubrics_rerun_exp_v5", "HELMInstruct_generic_rubrics_rerun_exp_v5"]
num_runs = 5

rubric_generator_name = "gpt4_CoT_v0"  # WARNING: DO NOT CHANGE THIS! We want to always use gpt4_CoT_v0, to make cross-project comparison easy.

# 12 categories total
category_to_num_examples = {
    "Information seeking":    249,
    "Creative Writing":       239,
    "Coding & Debugging":     104,
    "Reasoning":               51,
    "Editing":                 43,
    "Planning":                39,
    "Brainstorming":           35,
    "Math":                    34,
    "Role playing":            31,
    "Advice seeking":          14,
    "Data Analysis":           10,
    "Others":                   7,
}

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    instruction_set = "wildbench"

    for run_idx in range(num_runs):
        print(f"====================RUN {run_idx} START====================")
        for exp_suffix in exp_suffixes:
            print(f"====================EXP {exp_suffix} START====================")
            for evaluator_name in evaluator_names:
                # if run_idx > 0:
                #     # Clear evaluator cache to make sure 2nd run doesn't reuse 1st run's cache
                #     evaluator_cache_path = Path(f'helm_instruct/evaluator_configs/{evaluator_name}/annotations_seed0_configs.json')
                #     if evaluator_cache_path.exists() and evaluator_cache_path.is_file():
                #         evaluator_cache_path.unlink()
                for category in categories:
                    print(f"category: {category}")
                    n_max_examples = category_to_num_examples[category]
                    print(f"# examples: {n_max_examples}")
                    
                    # instructions = get_instructions(n_max_examples,
                    #                                             instruction_set=instruction_set,
                    #                                             category=category,
                    #                                             # sample_by_category=True,
                    #                                             with_additional_info=False,
                    #                                             n_to_print=0)
                    instructions_with_add_info = get_instructions(n_max_examples,
                                                                category=category,
                                                                instruction_set=instruction_set,
                                                                with_additional_info=True,
                                                                n_to_print=0)
                    
                    # ambiguous and overly complex instructions
                    # filtered_indices = [3, 5, 6, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23]
                    # instructions = [instructions[i] for i in range(len(instructions)) if i not in filtered_indices]
                    # instructions_with_add_info = [instructions_with_add_info[i] for i in range(len(instructions_with_add_info)) if i not in filtered_indices]

                    print(f"len(instructions_with_add_info): {len(instructions_with_add_info)}")

                    # df_criteria = get_criteria(instructions, n_to_print=0)
                    df_criteria_with_add_info = get_criteria(instructions_with_add_info, n_to_print=0)

                    print(f"len(df_criteria_with_add_info): {len(df_criteria_with_add_info)}")

                    def _clean_up_criteria_column(x):
                        if isinstance(x, float):
                            return None
                        else:
                            return [d['aspect'] for d in x]

                    for model_name in model_names:
                        # Rubric generation
                        df_rubrics = get_detailed_rubrics(df_criteria_with_add_info, n_to_print=0, is_store_missing_annotations=True, annotators_config=rubric_generator_name)
                        # I don't know why, but sometimes we can have duplicated "criteria" column :(
                        # Here we drop the 2nd column (hopefully that's the one that's all NaN)
                        if len(df_rubrics.columns[df_rubrics.columns.str.contains('criteria')]) > 1:
                            df_rubrics = df_rubrics.loc[:, ~df_rubrics.columns.duplicated()]

                        df_rubrics.criteria = df_rubrics.criteria.apply(_clean_up_criteria_column)

                        # Drop rows where the "scoring_scales" column is not a dict (because sometimes rubric generator will refuse to generate rubric)
                        mask = df_rubrics["scoring_scales"].apply(lambda x: isinstance(x, dict))
                        df_rubrics = df_rubrics[mask]

                        # Completion (by candidate model)
                        print(f"len(df_rubrics): {len(df_rubrics)}")
                        completions = get_model_completions(df_rubrics, model_name, n_to_print=0)
                        # print(f"completions: {completions}")

                        # Evaluation
                        completions = ae_utils.convert_to_dataframe(completions)
                        if "no_rubrics" in exp_suffix:
                            # Remove rubrics (i.e. use single criteria) to mimic AlpacaEval
                            completions['criteria'] = [['General Quality']] * len(completions)
                            new_rubric = {
                                'General Quality': {
                                    'Excellent': 'The response is highly relevant to the prompt, addressing all aspects comprehensively and staying on topic throughout. All information presented is accurate and well-supported by evidence or logical reasoning. The response is very clear and easy to understand, with a well-organized structure and logical flow of ideas. No extraneous information is included.',
                                    'Good': 'The response is generally relevant to the prompt, addressing most aspects and mostly staying on topic with minor deviations. Most information is accurate, with only minor errors or unsupported statements. The response is generally clear and understandable, mostly well-organized with a logical flow of ideas, though minor ambiguities may be present. Only minor extraneous information is included.',
                                    'Fair': 'The response is somewhat relevant to the prompt, addressing some aspects but not comprehensively, with significant deviations or off-topic information. There are several errors or unsupported statements, though some correct information is present. The response is somewhat clear but has noticeable issues with organization or flow of ideas, leading to some confusion. Some extraneous information is included.',
                                    'Poor': 'The response is minimally relevant or irrelevant to the prompt, failing to address key aspects and largely off-topic. It contains many errors and unsupported statements, with little to no correct information. The response is unclear and difficult to understand, poorly organized with a lack of logical flow. A lot of extraneous information is included.'
                                }
                            }
                            completions['detailed_analytic_rubric'] = [new_rubric] * len(completions)
                        elif "HELMInstruct_generic_rubrics" in exp_suffix:
                            # Use HELMInstruct generic rubrics
                            completions['criteria'] = [['Helpfulness', 'Understandability', 'Completeness', 'Conciseness', 'Harmlessness']] * len(completions)
                            new_rubric = {
                                'Helpfulness': {
                                    'Excellent': 'Provides brilliant ideas that address the task accurately',
                                    'Good': 'Is mostly relevant and seems very useful',
                                    'Fair': 'Is mostly relevant, but might not be super helpful',
                                    'Poor': 'Is only somewhat relevant',
                                },
                                'Understandability': {
                                    'Excellent': 'Is well-written and easy to understand',
                                    'Good': 'Is understandable but includes minor writing mistakes',
                                    'Fair': 'Is probably legible but difficult to comprehend',
                                    'Poor': 'Is mostly unreadable or difficult to comprehend',
                                },
                                'Completeness': {
                                    'Excellent': 'Gives a thorough and comprehensive answer',
                                    'Good': 'Provides a complete answer but could include more details or justifications',
                                    'Fair': 'Mostly answers the question but may miss some points',
                                    'Poor': 'Is only a partial answer and leaves out important information',
                                },
                                'Conciseness': {
                                    'Excellent': 'Is exceptionally concise, providing information efficiently',
                                    'Good': 'Is concise and to the point, with minimal unnecessary content',
                                    'Fair': 'Is moderately concise with minimal unnecessary content',
                                    'Poor': 'Is quite wordy',
                                },
                                'Harmlessness': {
                                    'Excellent': 'Is completely harmless and free of toxic content',
                                    'Good': 'Does not pose any harm but might have some slightly toxic content',
                                    'Fair': 'May include some potentially harmful or toxic content',
                                    'Poor': 'Is somewhat offensive, toxic, or promotes bad behaviors',
                                },
                            }
                            completions['detailed_analytic_rubric'] = [new_rubric] * len(completions)

                        evaluations = get_evaluations(completions, annotators_config=evaluator_name)
                        print(evaluations)
                        print(f"Category: {category}")
                        print(f"Model name: {model_name}")
                        print(f"Evaluator name: {evaluator_name}")
                        print(f"# evaluations: {len(evaluations)}")
                        print(f"mean of avg_score: {evaluations['avg_score'].mean()}")
                        print(f"STD of avg_score: {evaluations['avg_score'].std()}")
                        print("===========EVALUATION FINISHED============")
                        print()
                        # visualize_correct_rubric(evaluations, n_to_print=2)

                        # Save result to disk
                        output_path = Path(f'outputs/{exp_suffix}')
                        output_path.mkdir(parents=True, exist_ok=True)
                        with open(f'{output_path}/{category}___{model_name}___{evaluator_name}___completions_{run_idx}.pickle', 'wb') as file:
                            pickle.dump(completions, file)
                        with open(f'{output_path}/{category}___{model_name}___{evaluator_name}___evaluations_{run_idx}.pickle', 'wb') as file:
                            pickle.dump(evaluations, file)
            print(f"====================EXP {exp_suffix} DONE====================")
        print(f"====================RUN {run_idx} DONE====================")