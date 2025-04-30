import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_json(f"{BASE}/data/benchmark/rubriceval_general/wildbench_rubrics_w_add_info.json")
df = df[["id", "instruction", "category", "intent", "appropriate", "useful_info_to_eval_instruction"]]
df.to_json(f"{BASE}/data/benchmark/rubriceval_general/instructions.json")