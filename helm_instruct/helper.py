from helm_instruct import Instructionator, Rubricator, Completor, Evaluator
from alpaca_eval import utils as ae_utils
from alpaca_eval import constants as ae_const
import ast
import pandas as pd
import numpy as np
import yaml
from alpaca_eval.decoders import openai
from IPython.display import Markdown, display
def printmd(*args, is_replace_newline=False):
    string = " ".join([str(arg) for arg in args])
    if is_replace_newline:
        string = string.replace("\n", "<br>").replace("\t", "&emsp;")
    display(Markdown(string))
    
def get_mean_value(x):
    try:
        seq = list(ast.literal_eval(x).values())
    except:
        seq = [np.nan]
    return np.mean(seq)

def dict_reverser(d):
    return {v: k for k, v in d.items()}

def get_instructions(n_max_examples, is_use_alpacaeval_instructions=False, n_to_print=0):
    if is_use_alpacaeval_instructions:
        instructions = ae_const.ALPACAEVAL_REFERENCE_OUTPUTS_2()
        instructions = instructions.to_pandas().sample(n_max_examples, random_state=123)
        instructions["prompt"] = instructions["instruction"]
        instructions["category"] = instructions["dataset"]
        instructions = instructions.to_dict(orient="records")
    else:
        instructionator = Instructionator()
        n_categories = 11
        instructions = instructionator.generate_n_instructions_per_categories(max(n_max_examples // n_categories, 1))

    if n_to_print:
        for instruction in instructions[:n_to_print]:
            printmd("**Category**: ", instruction["category"])
            printmd("**Prompt**: ", instruction["prompt"])
            printmd("---------------------\n\n\n")

    return instructions

def get_rubrics(instructions, n_to_print=0):
    df_instructions = ae_utils.convert_to_dataframe(instructions)
    rubricator = Rubricator()
    rubrics = rubricator(df_instructions)
    df_rubrics = rubricator.make_df_rubrics(rubrics)

    if n_to_print:
        for i in range(min(len(df_rubrics), n_to_print)):

            printmd("**Example:** ", i)
            printmd("**Category:** ", df_rubrics.loc[i, "category"])
            printmd("\n**Prompt:** ", df_rubrics.loc[i, "final_prompt"])
            display(pd.DataFrame(df_rubrics.loc[i, "detailed_analytic_rubric"]).T)
            printmd("---------------------\n\n\n")

    return df_rubrics

def get_completions(rubrics, model_name, n_to_print=0):
    df_rubrics = ae_utils.convert_to_dataframe(rubrics)
    model = Completor(annotators_config=model_name)
    completions = pd.DataFrame(model(df_rubrics))
    completions = completions[completions["raw_completion"].notnull()].to_dict(orient="records")

    for completion in completions[:n_to_print]:
        printmd("**Category:** ", completion["category"])
        printmd("\n**Prompt:**\n", completion["final_prompt"])
        printmd("\n**Output:**\n", completion["output"])
        printmd("---------------------\n\n\n")

    return completions

def get_evaluations(completions, **kwargs):
    completions= ae_utils.convert_to_dataframe(completions)
    evaluator = Evaluator(**kwargs)
    scores = evaluator(completions)
    df_scores = ae_utils.convert_to_dataframe(scores).query("not output.isin(['', ' '])")
    if len(completions) != len(df_scores):
        printmd(f"Warning: {len(completions) - len(df_scores)} completions had to be dropped due to empty output.")
    df_scores["avg_score"] = df_scores["criteria_scores"].apply(lambda x: pd.Series(x).mean())
    printmd(completions.iloc[0]["annotator"], df_scores["avg_score"].mean())

    df_scores["criteria_annotations"] = df_scores.apply(lambda s: {k: dict_reverser(s["scoring_scales"])[v] for k, v in
                                                             s["criteria_scores"].items()}, axis=1)

    return df_scores


def apply_color(value, criteria, evaluation):
    if criteria in evaluation:
        if value == evaluation[criteria]:
            return 'background-color: orange'
    return ''


def color_df(df, to_color):
    # Create a temporary DataFrame to hold the styles
    styled_df = pd.DataFrame('', index=df.index, columns=df.columns)
    for index in df.index:
        for col in df.columns:
            # Apply color based on the rubric
            styled_df.loc[index, col] = apply_color(col, index, to_color)
    return styled_df

def visualize_correct_rubric(df_scores, n_to_print=5, is_sort_by_score=True):
    df_all = ae_utils.convert_to_dataframe(df_scores)
    if is_sort_by_score:
        df_all = df_all.sort_values("avg_score").reset_index(drop=True)
    for i in range(min(len(df_all), n_to_print)):
        printmd("**Example**: ", i)
        printmd("**Category**: ", df_all.loc[i, "category"])
        printmd("\n**Prompt**: ", df_all.loc[i, "final_prompt"])
        printmd("\n**Output**: ", df_all.loc[i, "output"])
        printmd("\n**Feedback**: ",  yaml.dump({f"<u>{k}</u>":v for k,v in df_all.loc[i, "feedback"].items()},
                                                        default_flow_style=False), is_replace_newline=True)
        df_rubric = pd.DataFrame(df_all.loc[i, "detailed_analytic_rubric"]).T
        styled_df = df_rubric.style.apply(color_df, to_color=df_all.loc[i, "criteria_annotations"], axis=None)
        display(styled_df)
        printmd("---------------------\n\n\n")


def summarize_results(df_scores):
    df_all = ae_utils.convert_to_dataframe(df_scores)
    instruction = ""
    for i, r in df_all.iterrows():

        instruction += f"""
# Example {i}

## Category
{r['category']}

## Prompt
{r['final_prompt']}

## Score per rubric
{yaml.dump({k: r['criteria_annotations'][k] + ". " + r['detailed_analytic_rubric'][k][v] for k, v in r['criteria_annotations'].items()},
           default_flow_style=False)}
           
## Expert feedback
{yaml.dump(r['feedback'], default_flow_style=False)}
"""

    n_to_eval = len(df_all)
    prompt = f"""
<|im_start|>system
You are a helpful assistant that summarizes the strengths and weaknesses of answers of a model. 
In particular we evaluated the answers of a model on {n_to_eval} instructions, and have scored each of them with a separate rubric. 
We now want a simple aggregate summary of what the model does well and bad. 
The summary should be very concise and use bullet point. 
The summary will be read by a user that wants to know whether they can use the model for their application.
The user doesn't know about the exact instructions we sued for evaluations.
<|im_end|>
<|im_start|>user
{instruction}
<|im_end|>
"""

    out = openai.openai_completions(
        prompts=[prompt],
        model_name="gpt-4-1106-preview",
        max_tokens=4096,
    )

    return out["completions"][0]
