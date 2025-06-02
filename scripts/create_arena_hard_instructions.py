import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw_questions = pd.read_json(f"{BASE}/data/arena_hard/instructions/question.jsonl", lines = True)
raw_questions["instruction"] = raw_questions["prompt"]
raw_questions = raw_questions.drop("prompt", axis = 1)
raw_questions.to_json(f"{BASE}/data/arena_hard/instructions/instructions.json")