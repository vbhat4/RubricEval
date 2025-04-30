from alpaca_eval import utils as ae_utils
from .evaluators import NaiveEvaluator, WildBenchEvaluator
import pandas as pd
import numpy as np
import logging
import json
import re


logger = logging.getLogger(__name__)


def evaluate_baseline(input_path, output_path, evaluator_configs):
    df_input = ae_utils.load_or_convert_to_dataframe(input_path)
    evaluator = NaiveEvaluator(annotators_config=evaluator_configs)
    df_evaluations = ae_utils.convert_to_dataframe(evaluator(df_input))

    df_evaluations.to_json(output_path, orient="records", indent=4)
    logger.info(f"Evaluation results are written to: {output_path}")
    return output_path


def evaluate_wildbench(input_path, output_path, evaluator_configs):
    df_input = ae_utils.load_or_convert_to_dataframe(input_path)
    evaluator = WildBenchEvaluator(annotators_config=evaluator_configs)
    df_evaluations = ae_utils.convert_to_dataframe(evaluator(df_input))

    df_evaluations.to_json(output_path, orient="records", indent=4)
    logger.info(f"Evaluation results are written to: {output_path}")
    return output_path


def extract_final_score(s):
    try:
        data = json.loads(s)
        return data.get("final_score", np.nan)
    except:
        return np.nan


def generate_report_baseline(input_path, model):
    df_input = ae_utils.load_or_convert_to_dataframe(input_path)
    df_input["final_score"] = df_input["evaluation"].apply(extract_final_score)
    df_input["final_score"] = np.where(df_input["output"] == "", np.nan, df_input["final_score"])

    weighted_mean = df_input["final_score"].mean()
    weighted_sem = df_input["final_score"].sem()

    report = {
        "model": model,
        "weighted_score_mean": weighted_mean,
        "weighted_score_sem": weighted_sem
    }
    
    report_json_path = "/".join(input_path.split("/")[:-1]) + "/" + "report_evaluations_evaluations.json"

    with open(report_json_path, "w") as file:
        json.dump(report, file)
    
    logger.info(f"Evaluation report is written to: {report_json_path} (raw).")
    return report_json_path


# Adapted from https://github.com/allenai/WildBench/blob/main/src/eval.py
def extract_values_from_json(json_string, keys = ["score", "Score", "rating", "Rating"], allow_no_quotes = True):
    extracted_values = {}
    for key in keys:
        if key not in json_string:
            continue
        # Create a regular expression pattern to find the value for the given key
        pattern = f'"{key}"\\s*:\\s*"([^"]*?)"'
        match = re.search(pattern, json_string)
        if match:
            extracted_values[key] = match.group(1)
        else:
            # Handle the case where the value might contain broken quotes
            pattern = f'"{key}"\\s*:\\s*"(.*?)"'
            match = re.search(pattern, json_string, re.DOTALL)
            if match:
                extracted_values[key] = match.group(1)
        if not match and allow_no_quotes:
            # to allow no quotes on the values
            pattern = f'"{key}"\\s*:\\s*([^,\\s]*)'
            match = re.search(pattern, json_string)
            if match:
                extracted_values[key] = match.group(1)
            else:
                # to allow no quotes on the keys
                pattern = f'{key}\\s*:\\s*([^,\\s]*)'
                match = re.search(pattern, json_string)
                if match:
                    extracted_values[key] = match.group(1)
    return extracted_values


# Adapted from https://github.com/allenai/WildBench/blob/main/src/eval.py
def parse_result(result_str, mode="json", eval_mode="score"): 
    assert eval_mode in ["score", "pairwise"]
    result_str = result_str.strip()
    result_str = result_str.replace("*", "")
    try: 
        try:
            parsed_result = json.loads(result_str)
            if "score" not in parsed_result and "Score" not in parsed_result and "rating" not in parsed_result and "Rating" not in parsed_result:
                return np.nan 
        except:
            parsed_result = extract_values_from_json(result_str)
            if "score" not in parsed_result and "Score" not in parsed_result and "rating" not in parsed_result and "Rating" not in parsed_result:
                return np.nan
    except Exception as e:
        return np.nan
    for key in ["score", "Score", "rating", "Rating"]:
        try:
            return int(parsed_result[key])
        except:
            pass
    return np.nan


def generate_report_wildbench(input_path, model):
    df_input = ae_utils.load_or_convert_to_dataframe(input_path)
    df_input["final_score"] = df_input["evaluation"].apply(parse_result)
    df_input["final_score"] = np.where(df_input["output"] == "", np.nan, df_input["final_score"])

    weighted_mean = df_input["final_score"].mean()
    weighted_sem = df_input["final_score"].sem()

    report = {
        "model": model,
        "weighted_score_mean": weighted_mean,
        "weighted_score_sem": weighted_sem
    }
    
    report_json_path = "/".join(input_path.split("/")[:-1]) + "/" + "report_evaluations_evaluations.json"

    with open(report_json_path, "w") as file:
        json.dump(report, file)
    
    logger.info(f"Evaluation report is written to: {report_json_path} (raw).")
    return report_json_path