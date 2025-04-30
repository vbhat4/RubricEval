import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from experiments.helpers import evaluate_wildbench, generate_report_wildbench
from rubric_eval.main import generate_outputs
import traceback


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = "wildbench_hard"
EVALUATOR = "gpt-4.1-2025-04-14"
MAX_WORKERS = 5
INSTRUCTIONS_PATH = f"{BASE}/data/benchmark/rubriceval_general/wildbench_base.json"
MODEL_CONFIGS_BASE_PATH = f"{BASE}/src/rubric_eval/configs/models_configs"
EVALUATOR_CONFIGS_PATH = f"{BASE}/scripts/configs/wildbench_evaluators_configs/{EVALUATOR}"
COMPLETIONS_BASE_PATH = f"{BASE}/data/{DATASET}/results/wildbench/annotated_completions"
EVALUATIONS_BASE_PATH = f"{BASE}/data/{DATASET}/results/wildbench/{EVALUATOR}_as_evaluator"


def setup():
    models = []

    for filename in os.listdir(MODEL_CONFIGS_BASE_PATH):
        if os.path.isdir(MODEL_CONFIGS_BASE_PATH + "/" + filename):
            models.append(filename)

    if not os.path.exists(COMPLETIONS_BASE_PATH):
        os.mkdir(COMPLETIONS_BASE_PATH)

    for model in models:
        if not os.path.exists(COMPLETIONS_BASE_PATH + "/" + model):
            os.mkdir(COMPLETIONS_BASE_PATH + "/" + model)

    if not os.path.exists(EVALUATIONS_BASE_PATH):
        os.mkdir(EVALUATIONS_BASE_PATH)

    for model in models:
        if not os.path.exists(EVALUATIONS_BASE_PATH + "/" + model):
            os.mkdir(EVALUATIONS_BASE_PATH + "/" + model)

    return models


def annotate_model_completions(model):
    if not os.path.exists(f"{COMPLETIONS_BASE_PATH}/{model}/completions.json"):
        generate_outputs(
            input_path = INSTRUCTIONS_PATH,
            output_path = f"{COMPLETIONS_BASE_PATH}/{model}/completions.json",
            model_configs = f"{MODEL_CONFIGS_BASE_PATH}/{model}"
        )


def evaluate_model(model):
    if not os.path.exists(f"{EVALUATIONS_BASE_PATH}/{model}/evaluations.json"):
        evaluate_wildbench(
            input_path = f"{COMPLETIONS_BASE_PATH}/{model}/completions.json",
            output_path = f"{EVALUATIONS_BASE_PATH}/{model}/evaluations.json",
            evaluator_configs = EVALUATOR_CONFIGS_PATH
        )


def generate_summary(model):
    if not os.path.exists(f"{EVALUATIONS_BASE_PATH}/{model}/report_evaluations_evaluations.json"):
        generate_report_wildbench(
            input_path = f"{EVALUATIONS_BASE_PATH}/{model}/evaluations.json",
            model = model
        )


def run_step(fn, models, step_name):
    with ThreadPoolExecutor(max_workers = MAX_WORKERS) as executor:
        futures = {executor.submit(fn, model): model for model in models}
        for future in as_completed(futures):
            model = futures[future]
            try:
                future.result()
                print(f"SUCCESS with model {model} on step {step_name}")
            except Exception as e:
                print(f"ERROR with model {model} on step {step_name}: {e!r}")
                traceback.print_exc()


def main():
    models = setup()
    run_step(annotate_model_completions, models, "annotate_model_completions")
    run_step(evaluate_model, models, "evaluate_models")
    run_step(generate_summary, models, "generate_summary")


if __name__ == "__main__":
    main()