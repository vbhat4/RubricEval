from helm_instruct import Instructionator, Rubricator, Completor, Evaluator, RubricBrainstormer, RubricGenerator
from typing import List, Dict, Any

import datasets
from alpaca_eval import utils as ae_utils
from alpaca_eval import constants as ae_const
import ast
import pandas as pd
import numpy as np
import yaml
from alpaca_eval.decoders import openai
from IPython.display import Markdown, display,HTML


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


"""
# TODO(stella):
1. What are the essential fields of the dataset that RubricEval needs?
- Once we define that, then we write adapters to translate known datasets into that format.
- I believe the essential fields are: prompt, category, additional_information(optional)
    - What's the expect format of additional_information? Right now it's:
    User intent: The user wants to make a report on project based learning focusing on the working of a remote using infrared rays
    Reference checklist:
    - Does the output clearly explain the concept of project-based learning?
    - Is there a detailed explanation of how remote controls use infrared rays to function?
    - Does the report include a step-by-step guide for a project on creating or demonstrating the working of a remote using infrared rays?
    - Are there clear objectives defined for the project-based learning activity?
    - Does the output include a list of materials needed for the project?
    - ...

2. What commands should the CLI support? (Does AlpacaEval have a CLI? Doesn't seem so.)
$ get_instructions -> this should take in dataset, process it (apply the dataset adapter), and print a few instructions
$ get_model_completions -> TODO
$ get_evaluations -> TODO
"""

def get_instructions(
    n_max_examples: int,
    category: str | None = None,
    instruction_set: str = "auto",
    sample_by_category: bool = False,
    with_additional_info: bool = False,
    n_to_print: int = 0,
    random_seed: int = 123,
) -> List[Dict[str, Any]]:
    # TODO(stella): need to refactor this function to make the dataset adapter code more generic. Also write more docs on this.
    if instruction_set == "auto":
        # TODO(stella): I don't 100% understand what's the generated instruction format in this branch yet. Need to investigate.
        instructionator = Instructionator()
        assert category is not None, "category must be provided if instruction_set is None"
        output = instructionator.generate_n_instruction_for_a_category(
            n=n_max_examples, category=category
        )
        instructions = ast.literal_eval(output[0]["categories_and_instructions"])
    elif instruction_set == "alpaca_eval_2":
        # TODO(stella): this branch doesn't seem to work: `BuilderConfig 'v1-legacy' not found. Available: ['alpaca_eval', 'alpaca_eval_gpt4_baseline', 'alpaca_eval_all_outputs', 'alpaca_farm_human_annotations', 'alpaca_farm_human_crossannotations', 'alpaca_eval_annotations_alpaca_eval_gpt4', 'alpaca_eval_annotations_claude']`
        instructions = ae_const.ALPACAEVAL_REFERENCE_OUTPUTS_2()
        instructions = instructions.to_pandas().sample(n_max_examples, random_state=random_seed)
        instructions["prompt"] = instructions["instruction"]
        instructions["category"] = instructions["dataset"]
        instructions = instructions.to_dict(orient="records")
    elif instruction_set == "wildbench":
        assert not (category is not None and sample_by_category)
        ds = datasets.load_dataset("allenai/WildBench")["test"]
        ds = ds.to_pandas()
        ds = ds[ds.apply(lambda x: len(x["conversation_input"]) == 1, axis=1)]  # maintain single-turn instructions for now
        instructions = pd.DataFrame()
        instructions["prompt"] = ds["conversation_input"].apply(lambda x: x[0]["content"])
        instructions["category"] = ds["primary_tag"]
        
        """
        Number of unique categories: 12

        Number of rows in each category:
        category
        Information seeking    249
        Creative Writing       239
        Coding & Debugging     104
        Reasoning               51
        Editing                 43
        Planning                39
        Brainstorming           35
        Math                    34
        Role playing            31
        Advice seeking          14
        Data Analysis           10
        Others                   7
        """

        if category:
            n_max_examples = min(n_max_examples, len(instructions[instructions["category"] == category]))
            instructions = instructions[instructions["category"] == category].sample(n_max_examples, random_state=random_seed)
        elif sample_by_category:
            num_per_category = n_max_examples // len(instructions["category"].unique())
            instructions = instructions.groupby("category").apply(lambda x: x.sample(num_per_category, random_state=random_seed)).reset_index(drop=True)
        else:
            instructions = instructions.sample(n_max_examples, random_state=random_seed)
        if with_additional_info:
            # merge the intent and checklist into the additional_information
            def _get_additional_information(x):
                info = ""
                info += f"User intent: {x['intent']}\n"
                checklist = '- ' + '\n- '.join(x['checklist'].tolist())
                info += f"Reference checklist:\n{checklist}\n"

                return info
            instructions["additional_information"] = ds.apply(_get_additional_information, axis=1)
        else:
            instructions["additional_information"] = "N/A"
        instructions = instructions.to_dict(orient="records")


    if n_to_print:
        print_instructions(instructions[:n_to_print])

    return instructions

def print_instructions(instructions, chunk_prompt_limit=1000):
    # TODO(stella): add doc: this is for Jupyter notebook use only and not Python console.
    for idx, instruction in enumerate(instructions):
        printmd("**Example**: ", idx)
        printmd("**Category**: ", instruction["category"])
        print_prompt(instruction["prompt"], chunk_limit=chunk_prompt_limit)
        if "intent" in instruction:   # wildbench
            printmd("**User intent**: ", instruction["intent"])
        if "additional_information" in instruction and instruction["additional_information"] != "N/A":
            printmd("**Additional information**: ", instruction["additional_information"])
        printmd("---------------------\n\n\n")


def print_prompt(prompt, chunk_limit=1000):
    half = chunk_limit // 2

    if len(prompt) > chunk_limit:
        printmd("**Prompt**: ", prompt[:half] + "\n\n***[... Omitted ...]***\n\n" + prompt[-half:])
    else:
        printmd("**Prompt**: ", prompt)



def get_criteria(instructions, n_to_print: int = 0, **annot_kwargs) -> pd.DataFrame:
    df_instructions = ae_utils.convert_to_dataframe(instructions)
    rubric_brainstormer = RubricBrainstormer(**annot_kwargs)
    criteria = rubric_brainstormer(df_instructions)
    df_criteria = rubric_brainstormer.make_df_rubrics(criteria)

    print_rubrics(df_criteria, n_to_print)

    return df_criteria

def get_detailed_rubrics(criteria, n_to_print: int = 0, **annot_kwargs) -> pd.DataFrame:
    df_criteria = ae_utils.convert_to_dataframe(criteria)
    rubric_generator = RubricGenerator(**annot_kwargs)
    detailed_rubrics = rubric_generator(df_criteria)
    df_detailed_rubrics = rubric_generator.make_df_rubrics(detailed_rubrics)

    print_rubrics(df_detailed_rubrics, n_to_print)

    return df_detailed_rubrics


def get_rubrics(instructions, n_to_print: int = 0) -> pd.DataFrame:
    df_instructions = ae_utils.convert_to_dataframe(instructions)
    rubricator = Rubricator()
    rubrics = rubricator(df_instructions)
    df_rubrics = rubricator.make_df_rubrics(rubrics)

    print_rubrics(df_rubrics, n_to_print)

    return df_rubrics


def print_rubrics(df_rubrics, n_to_print: int = 0, chunk_prompt_limit: int = 1000):
    if n_to_print:
        for i in range(min(len(df_rubrics), n_to_print)):
            printmd("**Example:** ", i)
            printmd("**Category:** ", df_rubrics.loc[i, "category"])
            print_prompt(df_rubrics.loc[i, "final_prompt"], chunk_limit=chunk_prompt_limit)
            if "clear_goals" in df_rubrics.columns:
                printmd("\n**Clear Goals:** ", df_rubrics.loc[i, "clear_goals"])
            if "criteria" in df_rubrics.columns:
                print_criteria(df_rubrics.loc[i, "criteria"])
            if "checklist" in df_rubrics.columns:  # wildbench
                printmd("\n**Checklist (Human)**:")
                display(pd.DataFrame(df_rubrics.loc[i, "checklist"], columns=['Checklist']))
            if "detailed_analytic_rubric" in df_rubrics.columns:
                printmd("\n**Detailed rubric**:")
                display(pd.DataFrame(df_rubrics.loc[i, "detailed_analytic_rubric"]).T)
            printmd("---------------------\n\n\n")


def print_criteria(criteria):
    df_criteria = pd.DataFrame(criteria)
    # checkist is a list, convert it to a string
    df_criteria["checklist"] = df_criteria["checklist"].apply(lambda x: "<br>".join(x))
    # df_criteria = df_criteria.map(lambda x: x.replace("\n", "<br>").replace("\t", "&emsp;"))
    html_criteria = df_criteria.to_html(escape=False)

    printmd("\n**Rubric**:")
    display(HTML(html_criteria))
    # display(df_criteria)


def get_model_completions(rubrics, model_name: str, n_to_print: int = 0):
    df_rubrics = ae_utils.convert_to_dataframe(rubrics)
    model = Completor(annotators_config=model_name)
    completions = pd.DataFrame(model(df_rubrics))
    completions = completions[completions["raw_completion"].notnull()].to_dict(
        orient="records"
    )

    for completion in completions[:n_to_print]:
        printmd("**Category:** ", completion["category"])
        printmd("\n**Prompt:**\n", completion["final_prompt"])
        printmd("\n**Output:**\n", completion["output"])
        printmd("---------------------\n\n\n")

    return completions


def get_evaluations(completions, **kwargs):
    pd.set_option('display.max_columns', None)
    
    evaluator = Evaluator(**kwargs)
    scores = evaluator(completions)
    df_scores = ae_utils.convert_to_dataframe(scores).query(
        "not output.isin(['', ' '])"
    )
    if len(completions) != len(df_scores):
        printmd(
            f"Warning: {len(completions) - len(df_scores)} completions had to be dropped due to empty output."
        )
    df_scores = df_scores.dropna(subset=['criteria_scores'])

    print(f"criteria_scores str type: \n{df_scores[df_scores['criteria_scores'].apply(lambda x: isinstance(x, str))]}")

    def calc_mean_with_exception_handling(x):
        x_value_set = set(x.values())
        if "N/A" in x_value_set or "n/a" in x_value_set:
            return None
        else:
            try:
                return pd.Series(x).mean()
            except:
                print(f"x: {x}")
                raise

    df_scores["avg_score"] = df_scores["criteria_scores"].apply(calc_mean_with_exception_handling)
    df_scores = df_scores[df_scores['avg_score'].notnull()]
    printmd(completions.iloc[0]["annotator"], df_scores["avg_score"].mean())

    def get_criteria_annotation(s):
        ret = {}
        for k, v in s["criteria_scores"].items():
            if v == 0 or v is None:
                ret[k] = None
            else:
                ret[k] = dict_reverser(s["scoring_scales"])[v]
        return ret
    
    df_criteria_annotations = df_scores.apply(
        get_criteria_annotation,
        axis=1,
    )
    assert not df_criteria_annotations.empty
    df_scores["criteria_annotations"] = df_criteria_annotations
    return df_scores


def apply_color(value, criteria, evaluation):
    if criteria in evaluation:
        if value == evaluation[criteria]:
            return "background-color: orange"
    return ""


def color_df(df, to_color):
    # Create a temporary DataFrame to hold the styles
    styled_df = pd.DataFrame("", index=df.index, columns=df.columns)
    for index in df.index:
        for col in df.columns:
            # Apply color based on the rubric
            styled_df.loc[index, col] = apply_color(col, index, to_color)
    return styled_df


def visualize_correct_rubric(df_scores, n_to_print=5, is_sort_by_score=True, chunk_prompt_limit=1000):
    df_all = ae_utils.convert_to_dataframe(df_scores)
    if is_sort_by_score:
        df_all = df_all.sort_values("avg_score").reset_index(drop=True)
    for i in range(min(len(df_all), n_to_print)):
        printmd("**Example**: ", i)
        printmd("**Category**: ", df_all.loc[i, "category"])
        print_prompt(df_all.loc[i, "final_prompt"], chunk_limit=chunk_prompt_limit)
        printmd("\n**Output**: ", df_all.loc[i, "output"])
        printmd(
            "\n**Feedback**: ",
            yaml.dump(
                {f"<u>{k}</u>": v for k, v in df_all.loc[i, "feedback"].items()},
                default_flow_style=False,
            ),
            is_replace_newline=True,
        )
        df_rubric = pd.DataFrame(df_all.loc[i, "detailed_analytic_rubric"]).T
        styled_df = df_rubric.style.apply(
            color_df, to_color=df_all.loc[i, "criteria_annotations"], axis=None
        )
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
