"""
Goal: show that if you can decompose evaluation into writing the gold decomposition (one time cost) and then evaluating, then you can make a bad model do the evaluation conditioned on the gold decomposition and it will be the same as if it was the main model

This experiment checks how useful decomposition is in general for evaluation. We do by comparing 4 methods:

1. No decomposition: direct evaluation conditioned on nothing. 
2. Decomposition based on solution: condition the evaluator on the gold solution.
3. Decomposition based on checklist: condition the evaluator on a gold checklist.
4. Decomposition based on rubric: condition the evaluator on a gold analytic rubric.

The goal here is not to show which is better but rather that 2/3/4 are better than 1. The experiment will be as follows:

Dataset: rubriceval_hard
Experiment 1 : all settings and all methods
    goal: in depth analysis with GPT4o and GPT4o-mini
    dataset: rubric_eval hard with outputs from mistral
    3 settings to run: 
        - GPT4o for gold and for evaluator
        - GPT4o-mini for gold and for evaluator
        - GPT4o for gold and GPT4o-mini for evaluator
    methods: the 4 (3 decomposition + 1 no decomposition)
    number of experiments: * 4 (methods) * 3 (settings) -1 = 11 => 550 examples
        -1 because no decomposition with for "GPT4o for gold and GPT4o-mini for evaluator" doesn't make sense
Experiment 2: 
    goal: show results are similar for different models
    dataset with outputs from mixtral
    3 settings to run: 
        - Sonnet 3.5 for gold and for evaluator
        - "Meta-Llama-3.1-8B-Instruct-Turbo" for gold and for evaluator
        - Sonnet 3.5 for gold and "Meta-Llama-3.1-8B-Instruct-Turbo" for evaluator
    methods: the rubric
    number of experiments: 3 (settings) => 150 examples
"""
import logging
from pathlib import Path
from typing import Optional

import fire
from alpaca_eval import utils as ae_utils
from evaluators import (
    BaseEvaluator,
    ChecklistEvaluator,
    ListRubricEvaluator,
    ListRubricSolutionEvaluator,
    NaiveEvaluator,
    RubricEvaluator,
    RubricSolutionEvaluator,
    SolutionEvaluator,
)

from rubric_eval import helpers, main


def run_dec_usefulness(
    experiment_name: str,
    gold_eval_models: list[tuple[str, str]],
    Evaluators: list[type[BaseEvaluator]],
    out_model: str = "Mixtral-8x22B-Instruct-v0.1",
    max_instances: Optional[int] = None,
    dataset: str = "rubriceval_sampled",
    output_kwargs: dict = {},
    preprocessor_kwargs: dict = {},
    eval_seed: int = 0,
    **evaluator_kwargs,
):
    benchmark_path = helpers.MAIN_DIR / f"data/benchmark/{dataset}/benchmark.json"
    if max_instances is None:
        sffx = ""
    else:
        sffx = f"_{max_instances}"
    dataset = dataset + sffx
    out_path = helpers.MAIN_DIR / f"data/benchmark/{dataset}/outputs_{out_model}.json"

    # get the mixtral outputs
    main.generate_outputs(
        model_configs=out_model,
        input_path=benchmark_path,
        output_path=out_path,
        is_rm_prev_columns=False,
        max_instances=max_instances,
        **output_kwargs,
    )

    for Evaluator in Evaluators:
        for gold_model, eval_model in gold_eval_models:
            if Evaluator == NaiveEvaluator and gold_model != eval_model:
                gold_model = "None"
            df = ae_utils.load_or_convert_to_dataframe(out_path)
            if max_instances:
                df = df.sample(max_instances, random_state=123)
            df_processed = Evaluator.preprocess(df, gold_model=gold_model, **preprocessor_kwargs)
            evaluator = Evaluator(annotators_config=eval_model, seed=eval_seed, **evaluator_kwargs)
            eval_results = evaluator(df_processed)
            df_eval = ae_utils.convert_to_dataframe(eval_results)
            df_graded = evaluator.postprocess(df_eval)
            res_path = f"results/exp_{experiment_name}/data_{dataset}/out_{out_model}/mode_{Evaluator.__name__}_gold_{gold_model}_eval_{eval_model}_eval-seed_{eval_seed}.json"
            res_path = Path(res_path)
            res_path.parent.mkdir(parents=True, exist_ok=True)
            df_graded.to_json(res_path, orient="records", indent=2)


def exp_decompositions_all(dataset: str = "rubriceval_sampled", **kwargs):
    gold_eval_models = [
        ("gpt-4o-2024-08-06_CoT_v0", "gpt-4o-2024-08-06_CoT_v0"),
        ("gpt-4o-mini-2024-07-18_CoT_v0", "gpt-4o-mini-2024-07-18_CoT_v0"),
        ("gpt-4o-2024-08-06_CoT_v0", "gpt-4o-mini-2024-07-18_CoT_v0"),
        # ("Qwen1.5-7B-Chat", "Qwen1.5-7B-Chat"),
        # ("gpt-4o-2024-08-06_CoT_v0", "Qwen1.5-7B-Chat"),
    ]

    run_dec_usefulness(
        experiment_name="decompositions_all",
        gold_eval_models=gold_eval_models,
        Evaluators=[
            ListRubricEvaluator,
            ListRubricSolutionEvaluator,
        ],  # [NaiveEvaluator, ChecklistEvaluator, RubricEvaluator, SolutionEvaluator, RubricSolutionEvaluator],
        **kwargs,
    )


def exp_decompositions_all_v1(dataset: str = "rubriceval_sampled", **kwargs):
    gold_eval_models = [
        ("gpt-4o-2024-08-06_CoT_v1", "gpt-4o-2024-08-06_CoT_v1"),
        ("gpt-4o-mini-2024-07-18_CoT_v1", "gpt-4o-mini-2024-07-18_CoT_v1"),
        ("gpt-4o-2024-08-06_CoT_v1", "gpt-4o-mini-2024-07-18_CoT_v1"),
    ]
    run_dec_usefulness(
        experiment_name="decompositions_all_v1",
        gold_eval_models=gold_eval_models,
        Evaluators=[RubricEvaluator],
        **kwargs,
    )


def exp_decompositions_other(dataset: str = "rubriceval_sampled", **kwargs):
    # TODO: write the rubric and evalautor for this experiment. the issue is that they do not use function calling
    gold_eval_models = [
        ("claude-3-5-sonnet-20240620_CoT_v0", "claude-3-5-sonnet-20240620_CoT_v0"),
        ("claude-3-haiku-20240307_CoT_v0", "claude-3-haiku-20240307_CoT_v0"),
        ("claude-3-5-sonnet-20240620_CoT_v0", "claude-3-haiku-20240307_CoT_v0"),
        # ("Meta-Llama-3.1-8B-Instruct-Turbo", "Meta-Llama-3.1-8B-Instruct-Turbo"),
        # ("claude-3-5-sonnet-20240620", "Meta-Llama-3.1-8B-Instruct-Turbo"),
    ]
    run_dec_usefulness(
        experiment_name="decompositions_other",
        gold_eval_models=gold_eval_models,
        Evaluators=[RubricEvaluator],
        **kwargs,
    )


def exp_variance_models(dataset: str = "rubriceval_sampled", **kwargs):
    gold_eval_models = [
        # ("gpt-4o-2024-08-06_CoT_v0", "Mixtral-8x7B-Instruct-v0.1_CoT_v0"),
        # ("gpt-4o-2024-08-06_CoT_v0", "Qwen2-72B-Instruct_CoT_v0"), # not allowed by Together.ai
        # ("gpt-4o-2024-08-06_CoT_v0", "Meta-Llama-3.1-8B-Instruct-Turbo_CoT_v0"),
        ("gpt-4o-2024-08-06_CoT_v0", "gemini-1.5-flash_CoT_v0"),
        ("gpt-4o-2024-08-06_CoT_v0", "gpt-3.5-turbo-0125_CoT_v0"),
        ("gpt-4o-2024-08-06_CoT_v0", "claude-3-5-sonnet-20240620_CoT_v0"),
        ("gpt-4o-2024-08-06_CoT_v0", "claude-3-haiku-20240307_CoT_v0"),
        ("gpt-4o-2024-08-06_CoT_v0", "gpt-4o-mini-2024-07-18_CoT_v0"),
        ("gpt-4o-2024-08-06_CoT_v0", "gpt-4o-2024-08-06_CoT_v0"),
        ("gpt-4o-2024-08-06_CoT_v0", "gemini-1.5-pro_CoT_v0"),
    ]
    run_dec_usefulness(
        experiment_name="variance_models",
        gold_eval_models=gold_eval_models,
        Evaluators=[NaiveEvaluator, ChecklistEvaluator, RubricEvaluator, SolutionEvaluator, RubricSolutionEvaluator],
        **kwargs,
    )


def exp_variance_sampling(dataset: str = "rubriceval_sampled", n_seeds: int = 5, **kwargs):
    gold_eval_models = [("gpt-4o-2024-08-06_CoT_v0", "gpt-4o-mini-2024-07-18_CoT_v0")]
    for seed in range(n_seeds):
        run_dec_usefulness(
            experiment_name="variance_sampling",
            gold_eval_models=gold_eval_models,
            Evaluators=[
                NaiveEvaluator,
                ChecklistEvaluator,
                RubricEvaluator,
                SolutionEvaluator,
                RubricSolutionEvaluator,
            ],
            eval_seed=seed,
            **kwargs,
        )


if __name__ == "__main__":
    fire.Fire(
        {
            "decompositions_all": exp_decompositions_all,
            "decompositions_all_v1": exp_decompositions_all_v1,
            "decompositions_other": exp_decompositions_other,
            "variance_models": exp_variance_models,
            "variance_sampling": exp_variance_sampling,
        }
    )
