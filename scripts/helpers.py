import json
import re
from pathlib import Path

import constants
import numpy as np
import pandas as pd
from alpaca_eval import utils as ae_utils


def load_readable_stanford_ml_dataset(path: Path, assert_constants: bool = True):
    dataset = []
    for f_path in path.glob("*.md"):
        with open(f_path, "r") as f:
            expert_txt = f.read()
            expert_dict = convert_md_to_dict(expert_txt)
            expert_dict["short_name"] = f_path.stem
            dataset.append(expert_dict)
    df = pd.DataFrame(dataset)
    if assert_constants:
        assert set(df["short_name"].unique()) == set(constants.STANFORD_ML_NAMES)
    for c in df.columns:
        if "_time_sec" in c and df[c].dtype in [str, object]:
            df[c] = df[c].replace("", np.nan).astype(float)
    return df


def convert_dict_to_markdown(d: dict):
    md = ""
    for k, v in d.items():
        if isinstance(v, (list, tuple, dict)):
            v = json.dumps(v, indent=2)
        md += f"# <{k}>:\n{v}\n"
    return md


def convert_md_to_dict(md: str):
    d = {}
    pattern = r"# <(.*?)>:\n"
    matches = list(re.finditer(pattern, md))
    for i, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i < len(matches) - 1 else len(md)
        value = md[start:end].rstrip("\n")
        value = ae_utils.convert_str_to_sequence(value)
        d[key] = value
    return d
