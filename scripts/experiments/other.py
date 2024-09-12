import copy
import logging
import sys
from pathlib import Path

import fire
import pandas as pd

sys.path.append(str(Path(__file__).parents[1]))
import constants
import helpers


def _load_stanford_ml_expert_responses(dir_path: Path):
    df = helpers.load_readable_stanford_ml_dataset(dir_path)
    new_rows = []
    for i, row in df.iterrows():
        init_row = dict(category=row["category"], instruction=row["instruction"], name=row["short_name"])
        for i in range(1, 100):
            if f"rank_{i}" not in df.columns:
                break
            new_row = copy.deepcopy(init_row)
            new_row["rank"] = row[f"rank_{i}"]
            new_row["feedback"] = row[f"feedback_{i}"]
            new_row["response"] = row[f"response_{i}"]
            new_rows.append(new_row)
        # add the expert solution as rank 0
        new_row = copy.deepcopy(init_row)
        new_row["rank"] = 0
        new_row["response"] = row["expert_solution"]
        new_rows.append(new_row)
    df = pd.DataFrame(new_rows)
    return df


def _load_all_stanford_ml_expert_responses():
    all_dfs = []
    for error_type in constants.STANFORD_ML_ERROR_TYPES:
        df = _load_stanford_ml_expert_responses(constants.STANFORD_ML_ERROR_DIR / error_type)
        df["error_type"] = error_type
        logging.info(f"Loaded {len(df)} responses for {error_type}")
        all_dfs.append(df)
    df = pd.concat(all_dfs, axis=0)
    df["rank"] = df["rank"].astype(int)
    return df


def make_stanford_ml_general_ranking():
    df_expert = _load_all_stanford_ml_expert_responses()
    rank = df_expert["rank"].max() + 1

    # load all the responses from different LLM, as ranked by LB_MODELS and concatenate them as next rank
    all_model_outputs = []
    for i, model in enumerate(constants.LB_MODELS):
        df_model = pd.read_json(constants.STANFORD_ML_DIR / f"outputs_{model}.json")
        df_model["rank"] = rank + i
        df_model["response"] = df_model["output"]
        all_model_outputs.append(df_model)

    df = pd.concat([df_expert] + all_model_outputs, axis=0)
    df.sort_values(by=["instruction", "rank"], inplace=True)
    for inst in df["instruction"].unique():
        dict_to_save = {}
        cols_to_keep_i = ["response", "feedback", "model"]
        cols_to_keep = ["category", "instruction", "name", "rank"] + cols_to_keep_i
        df_name = df.loc[df["instruction"] == inst, cols_to_keep].drop_duplicates()
        for i in range(1, len(df_name) + 1):
            row = df_name.iloc[i - 1]
            if i == 1:
                dict_to_save["category"] = row["category"]
                dict_to_save["instruction"] = row["instruction"]
            for c in cols_to_keep_i:
                dict_to_save[f"{c}_{i}"] = row[c]
            dict_to_save[f"rank_{i}"] = i
        readable_dataset = helpers.convert_dict_to_markdown(dict_to_save)
        name = next((n for n in df_name["name"].unique() if pd.notna(n)), None)
        assert name is not None

        # save to markdown file
        general_dir = constants.STANFORD_ML_ERROR_DIR / "general"
        general_dir.mkdir(parents=True, exist_ok=True)
        with open(general_dir / f"{name}.md", "w") as file:
            file.write(readable_dataset)


if __name__ == "__main__":
    fire.Fire(
        {
            "stanford_ml_general_ranking": make_stanford_ml_general_ranking,
        }
    )
