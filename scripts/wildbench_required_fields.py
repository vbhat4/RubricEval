import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Adapted from https://github.com/allenai/WildBench/blob/main/src/eval.py
def compose_eval_item(b, histories, last_queries, checklists):
    history = ""
    checklist = b["checklist"]
    if len(b["conversation_input"]) > 0:
        for x in b["conversation_input"][:-1]:
            if x["role"] == "user":
                history += "USER: " + x["content"] + "\n\n"
            elif x["role"] == "assistant":
                history += "ASSISTANT: " + x["content"] + "\n\n"
    last_query = b["conversation_input"][-1]["content"]
    histories.append(history)
    last_queries.append(last_query)
    checklists.append(checklist)

df = pd.read_json(f"{BASE}/data/benchmark/rubriceval_general/wildbench_rubrics_w_add_info.json")
df = df[["id", "session_id", "instruction", "conversation_input", "category", "checklist", "intent", "appropriate"]]

histories = []
last_queries = []
checklists = []
for _, row in df.iterrows():
    compose_eval_item(row, histories, last_queries, checklists)
df["history"] = histories
df["user_query"] = last_queries
df["checklist"] = checklists

df.to_json(f"{BASE}/data/benchmark/rubriceval_general/wildbench_base.json")