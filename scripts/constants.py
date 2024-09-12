from rubric_eval import helpers as re_helpers

LB_MODELS = [
    "gpt-4o-2024-08-06",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "Meta-Llama-3.1-405B-Instruct-Turbo",
    "Meta-Llama-3.1-70B-Instruct-Turbo",
    "Qwen2-72B-Instruct",
    "claude-3-haiku-20240307",
    "Meta-Llama-3.1-8B-Instruct-Turbo",
    "Meta-Llama-3-8B-Instruct-Turbo",
    "gpt-4o-mini-2024-07-18",
]

STANFORD_ML_NAMES = [
    "backprop",
    "information",
    "least_squares",
    "optimization",
    "quantile",
    "rl",
    "tokenization",
    "transformer",
    "variance",
]
STANFORD_ML_ERROR_TYPES = ["style", "reasoning", "instruction_following", "factuality"]

# PATHS
MAIN_DIR = re_helpers.MAIN_DIR
BENCHMARK_DIR = MAIN_DIR / "data/benchmark"
STANFORD_ML_DIR = BENCHMARK_DIR / "rubriceval_stanford_ml"
STANFORD_ML_ERROR_DIR = STANFORD_ML_DIR / "responses"
