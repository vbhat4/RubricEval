"""
python extract_data_from_log.py output1_snapshot.txt
"""

import sys
from collections import OrderedDict

def save_data_to_csv(exp_to_category_to_model_to_score_tuple, filename):
    with open(filename, mode='w') as file:
        for exp_suffix in exp_to_category_to_model_to_score_tuple:
            file.write(f"Exp: {exp_suffix}\n")
            category_to_model_to_score_tuple = exp_to_category_to_model_to_score_tuple[exp_suffix]
            for category in category_to_model_to_score_tuple:
                model_name_row = []
                mean_score_row = []
                std_score_row = []
                for model_name in category_to_model_to_score_tuple[category]:
                    model_name_row.append(model_name)
                    mean_score, std_score = category_to_model_to_score_tuple[category][model_name]
                    mean_score_row.append(mean_score)
                    std_score_row.append(std_score)
                file.write(f"Category: {category}\n")
                file.write(",".join(model_name_row) + "\n")
                file.write(",".join(mean_score_row) + "\n")
                file.write(",".join(std_score_row) + "\n")
                file.write("\n")


"""
Category: Advice seeking
Model name: Meta-Llama-3-8B-Instruct
Evaluator name: gpt-4o-2024-08-06_CoT_v0
# evaluations: 13
mean of avg_score: 2.902564102564102
STD of avg_score: 0.8646837934632884
"""


exp_to_category_to_model_to_score_tuple = OrderedDict()
exp_suffix = None
category = None
model_name = None
mean_score = None
std_score = None
with open(sys.argv[1], 'r') as log_file:
    for line in log_file.readlines():
        line = line.strip()
        if line.startswith("====================EXP ") and line.endswith(" START===================="):
            exp_suffix = line.replace("====================EXP ", "").replace(" START====================", "")
        if line.startswith("Category: "):
            category = line.replace("Category: ", "")
        if line.startswith("Model name: "):
            model_name = line.replace("Model name: ", "")
        if line.startswith("mean of avg_score: "):
            mean_score = line.replace("mean of avg_score: ", "")
        if line.startswith("STD of avg_score: "):
            std_score = line.replace("STD of avg_score: ", "")
        if mean_score is not None and std_score is not None:
            assert exp_suffix is not None
            if exp_suffix not in exp_to_category_to_model_to_score_tuple:
                exp_to_category_to_model_to_score_tuple[exp_suffix] = OrderedDict()
            assert category is not None
            if category not in exp_to_category_to_model_to_score_tuple[exp_suffix]:
                exp_to_category_to_model_to_score_tuple[exp_suffix][category] = OrderedDict()
            exp_to_category_to_model_to_score_tuple[exp_suffix][category][model_name] = (mean_score, std_score)
            category = None
            model_name = None
            mean_score = None
            std_score = None
        if line.startswith("====================EXP ") and line.endswith(" DONE===================="):
            exp_suffix = None
        if line.startswith("====================RUN ") and line.endswith(" DONE===================="):
            run_idx = int(line.replace("====================RUN ", "").replace(" DONE====================", ""))
            save_data_to_csv(exp_to_category_to_model_to_score_tuple, f"run_{run_idx}.csv")
            exp_to_category_to_model_to_score_tuple.clear()

        