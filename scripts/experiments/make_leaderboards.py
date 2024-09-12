from typing import Optional, Sequence

import fire
from constants import LB_MODELS, MAIN_DIR

from rubric_eval import helpers, main


def run_leaderboard(
    experiment_name: str,
    # Evaluator: list[type[BaseEvaluator]],
    out_models: Sequence[str] = "Mixtral-8x22B-Instruct-v0.1",
    max_instances: Optional[int] = None,
    dataset: str = "stanford_ml",
    output_kwargs: dict = {}
    # **evaluator_kwargs,
):
    benchmark_path = MAIN_DIR / f"data/benchmark/{dataset}/benchmark.json"
    if max_instances is None:
        sffx = ""
    else:
        sffx = f"_{max_instances}"
    dataset = dataset + sffx
    for out_model in out_models:
        out_path = MAIN_DIR / f"data/benchmark/{dataset}/outputs_{out_model}.json"

        # get the mixtral outputs
        main.generate_outputs(
            model_configs=out_model,
            input_path=benchmark_path,
            output_path=out_path,
            is_rm_prev_columns=False,
            max_instances=max_instances,
            **output_kwargs,
        )


def leaderboard_stanford_ml(
    max_instances: Optional[int] = None,
    output_kwargs: dict = {},
):
    run_leaderboard(
        experiment_name="leaderboard_stanford_ml",
        out_models=LB_MODELS,
        max_instances=max_instances,
        dataset="rubriceval_stanford_ml",
        output_kwargs=output_kwargs,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "stanford_ml": leaderboard_stanford_ml,
        }
    )
