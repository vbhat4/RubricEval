# <a href="https://huggingface.co/spaces/vbhat4/rubriceval" target="_blank"><img src="https://raw.githubusercontent.com/YannDubs/RubricEval/main/docs/rubriceval_icon.png" width="35"></a> [RubricEval](https://huggingface.co/spaces/vbhat4/rubriceval): Scalable Expert Evaluation of Language Model

[![License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/RubricEval/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

---

**RubricEval** scales expert evaluation of LLMs by combining expert-created rubrics with LLM-based auto-evaluation. Experts design detailed, example-specific rubrics (a one-time cost), which LLMs then apply at **scale**. Evaluating based on rubrics offers a more **interpretable** and **trustworthy** evaluation compared to the common practice of simply asking the model if an answer is "good" or "better" than another.

[add figure or video here]

---

<details open>
  <summary><b>Table of Contents</b></summary>

1. [Quick Start](#quick-start)
2. [RubricEval](#RubricEval)
   - [Overview](#overview) 
   - [Pipeline](#pipeline)
2. [Interpreting results](#interpreting-results)
3. [Use-cases](#use-cases)
    - [Evaluating a model](#evaluating-a-model)
    - [Applying RubricEval on your instructions](#applying-rubriceval-on-your-instructions)
4. [Contributing](#contributing)
    - [Contributing a model](#contributing-a-model)
    - [Contributing an eval set](#contributing-an-eval-set)
    - [Contributing an evaluator](#contributing-an-evaluator)
5. [Limitations](#limitations)
7. [Citation](#citation)
8. [Additional information](#additional-information)
   - [Code organization](#code-organization)
   - [Analysis](#additional-analysis-and-plots)
   - [Data release](#data-release)
   - [Related work](#related-work)
   - [Major updates](#major-updates)

</details>


# Quick Start

To install the stable release, run

```bash
pip install rubric-eval
```

<details>
  <summary>Nighlty installation</summary>

```bash
pip install git+https://github.com/tatsu-lab/rubric_eval
```

</details>

To evaluate a completions on the RubricEval benchmark, run the following two steps:

1. **Generate completions on the RubricEval instructions**: create a file `completions.json` that is a [JSON file of the RubricEval evaluation set](...) with an additional `completions` column containing the completions of the model to evaluate. You can use you favorite generation pipeline, or our pipeline based on [AlpacaEval](): 

```bash
rubric_eval get_completions \
--output_path=completions.json \
--model_config=<model_to_evaluate> # e.g. gpt-4o-2024-05-13
```

2. **Evaluate the completions**: evaluate the previously generated completions using the RubricEval LLM-judge. 


```bash
export OPENAI_API_KEY=<your_api_key> # for more complex configs, e.g. using Azure or switching clients see https://github.com/tatsu-lab/alpaca_eval/tree/main/client_configs/README.md 
rubric_eval --input_path=completions.json 
```

This will print the leaderboard in the terminal and generate an `evaluation_report.md` file at [...].



Both previous steps can be done in one step by running `
rubric_eval --model_config=<model_to_evaluate>`

See the following for more details about:
- [evaluating a model](#evaluating-a-model)
- [adding a model to the leaderboard](#contributing-a-model)
- [running the RubricEval pipeline](#applying-rubriceval-on-your-instructions) (including rubric creation) on your instructions

[//]: # (- interpreting RubricEval's results, see [here]&#40;#leaderboards-and-how-to-interpret-them&#41;)



# RubricEval 

## Overview

Current evaluation of LLMs in open-ended tasks involves asking an annotator (human or LLM) whether an answer is “good” or “better” than another. These criteria are often ill-defined making it hard to interpret and trust. The lack of trust is exacerbated when using LLM-judge, which may latch onto superficial features.  

**RubricEval** addresses this issue by providing annotators with detailed, example-specific rubrics that clearly define what constitutes a good answer. Experts create these rubrics (a one-time cost), and LLMs apply them at scale, resulting in an overall evaluation methodology that is interpretable, trustworthy, and scalable.

We provide the following:

- [**Leaderboard**](https://tatsu-lab.github.io/alpaca_eval/): a leaderboard of common models evaluated using RubricEval on the RubricEval dataset. To add your model to the leaderboard, see [here](#contributing-a-model).
- [**Toolkit for building RubricEval benchmark**](#applying-rubriceval-on-your-instructions): a simple interface for using RubricEval on your own instructions. We provide the code to (1) generate rubrics based on your rubrics and (optionally) expert inputs; and to (2) evaluate models based on these rubrics.
- [**RubricEval dataset**](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_eval.json): a set of X instructions with detailed rubrics. The instructions were filtered from the [WildBench](https://huggingface.co/datasets/allenai/WildBench) dataset for complexity. [Details here](#data-release).
- [**Human evaluation data**](#data-release): ...


## Pipeline

As shown in the above figure, the RubricEval evaluation pipeline can be decomposed into five main steps: instruction writing, rubric creation, completion, evaluation, and evaluation report.
The first two steps correspond to "[creating the benchmark]()", while the final three steps correspond to "[evaluating the benchmark]()".
Our code assumes that the input and output of each step is a JSON file with a list of instruction-specific dictionaries.
We now discuss each step in more detail.

1. **Instruction writing** the first step for creating any benchmark is collecting the instructions that we will evaluate the models on. This is not different from any other benchmark but note that the RubricEval pipeline is particularly well suited for high-stakes domains where (1) trust in the benchmark is crucial; but (2) expert are expensive. E.g. of domains that are well suited: law, medical, finance, etc.
For better interpretability, instructions may also be categorized by "category".
Note that the quality, diversity and complexity of instructions is more important than the quantity. Having as little as 50-100 good instructions are likely enough to evaluate well a model  

   As an example, for the RubricEval benchmark, we use a set of 391 challenging instructions [filtered](#data-release) from the WildBench dataset.


2. **Rubric creation** the new and crucial step required for a RubricEval benchmark is creating high-quality and detailed instruction-specific rubrics. A rubric is a table with rows corresponding to instruction-specific axes of evaluation and columns corresponding to instruction-agnostic levels of performance. The cells of the table are filled with detailed descriptions of what outputs fall in each category.

   For example, a rubric for ``write a Python function of the most efficient algorithm for primality test`` could have rows such as "clarification", "correctness", "efficiency" and columns "excellent", "fair", "poor". The cell for "clarification" and "excellent" may require the outputs to ask whether the primality test needs to be exact (vs probabilistic) and what range of numbers it needs to be the most efficient for. "correctness" and "fair" could be for functions that recognize most primes but fail on common edge cases such as 0, 1, and negative numbers. 

   Our codebase assumes that the instructions and rubrics are in a JSON file with the following format:

   ```json
   [
      {"instruction":  "<instruction to evaluate on>", 
       "category": "<[optional] category of the instruction>"
       "rubric": { // instruction-specific rubric
         "<axes 1>": { // first row of the rubric 
           "<1st level of performance>": "<description>", // column
           "<2nd level of performance>": "<description>",
           ... // more instruction-agnostic levels of performance
         },
         "<axes 2>": {
           "<1st level of performance>": "<description>",
           "<2nd level of performance>": "<description>",
           ...
         },
         ... // more instruction-specific axes of evaluation
       },
       ... // optional unused fields
     },
     ... // more instructions
   ]
   ```
   Such instruction-specific rubrics can be written directly by experts but this can be time consuming for (human) experts. 
   For more efficient rubric creation, we suggest and provide the code for the following LLM and expert collaborative pipeline for each instruction:
   1. **Checklist** experts write a high-level unstructured checklist of things that should be considered when evaluating as seen below.
   
      [add figure]
   
      In our codebase, this requires adding a column `additional_information` to the instructions file with the unstructured checklist. 
      For the  
   
   2. **Rubric brainstorming** an LLM converts the checklist to a high-level structured rubric. Experts can then review and modify the brainstormed rubric. E.g. below
   
      ![rubric_brainstorming.png](figures%2Frubric_brainstorming.png)

        In our codebase, this requires running the `RubricBrainstormer` annotator on the instructions file.
      
   
   3. **Rubric generation** the LLM generates the detailed rubric. Experts can then review and modify the generated rubric.


    2. **Generating the rubrics** given the high-level rubric, we can generate the final rubric. This is done by the `RubricGenerator` which is a simple LLM that takes in the high-level rubric and the instruction and generates the final rubric. This step should be simple for LLMs but tedious for humans. I would thus suggest always having an LLM do this step, with optional human review & modification.
3. 
4. pipeline is creating the rubrics. This is done in two steps:
    1. **Brainstorming the rubrics** given the instruction and additional information, an LLM or an expert could write some high-level rubric as seen below.
    2. **Generating the rubrics** given the high-level rubric, we can generate the final rubric. This is done by the `RubricGenerator` which is a simple LLM that takes in the high-level rubric and the instruction and generates the final rubric. This step should be simple for LLMs but tedious for humans. I would thus suggest always having an LLM do this step, with optional human review & modification.


3. given the instructions, an LLM or an expert could write some high-level rubric as seen below.

2. **Brainstorming the rubrics** given the instruction and additional information, an LLM or an expert could write some high-level rubric as seen below.
![rubric_brainstorming.png](figures%2Frubric_brainstorming.png)
    - **Real-world** in the real-world the additional cost of having experts write the high-level rubrics is minimal compared to adding a checklist (the only difference is that it has to be structured by different axes. Our belief is that having expert-written high-level rubrics would give much more realistic final rubrics and thus be worth the cost.
    - **Benchmark** for the benchmark, we provide a `RubricBrainstormer` that aims to convert the usntructured additional_information (e.g. checklist) to the high-level rubric. This is to allow converting any dataset to our format and should be used for example with WildBench.
3. **Generating the rubrics** given the high-level rubric, we can generate the final rubric. This is done by the `RubricGenerator` which is a simple LLM that takes in the high-level rubric and the instruction and generates the final rubric. This step should be simple for LLMs but tedious for humans. I would thus suggest always having an LLM do this step, with optional human review & modification. 
![rubric_generating.png](figures%2Frubric_generating.png)
4. **Decoding/Inference of the models** there's nothing new here, we just need to predict what different models would say. This is done by the `Completor` and `alpaca_eval` essentially already provides the code to evaluate hundreds of models.
5. **Evaluation** the `Evaluator` then takes in the predictions and the rubric and outputs some feedback, the selected category in the rubric, and the final score. This should be done only by LLM for scalability. This will likely be the hardest for the model in some domains (e.g. if it's bad at math it will be hard for it to evaluate math instructions). But for many valuable domains (healthcare, legal, ...)  I'm confident that current models can do a good job.
6. **Summarizer**  One of the main benefits of RubricEval is that it can provide more finegrained interpretability into what the model does well/wrong and why it was given a certain score. THe problem is that no one will read N>=200 feedback and selected rubrics. We would thus need a summarizer that takes in the feedback and selected rubric and generates a summary / model card of what the model does well and badly. This is a bit different from the other steps in that the function is many-to-one (many feedbacks to one summary), which is why we don't have a minimal class that inherits from AlpacaEval to do that. A minimal version of the summarizer is `summarize_results` but this should really be rewritten. Note that providing all the outputs & instructions & feedback & rubrics will probably exceed the context length of the llm judge. So I would suggest having a two (or more) layer summarizier pipeline, which recursively applies summarizers to the output of the previous summarizer. Note that the summarizier will need to be prompt tuned to ensure that it provides detailed and actionable summaries of the model's performance that are self contained (i.e. the human shouldn't need to know about the instructions to understand the summary).

----------

Then you can use it as follows:

```bash
export OPENAI_API_KEY=<your_api_key> # for more complex configs, e.g. using Azure or switching clients see https://github.com/tatsu-lab/alpaca_eval/tree/main/client_configs/README.md 

# This will generate an example instructions.json
python scripts/create_example_instructions.py instructions.json

# Generate rubrics based on provided instructions
rubric_eval get_rubrics \
--input_path=instructions.json \
--output_path=instructions_with_rubrics.json

# Generate completions based on provided instructions
rubric_eval get_completions \
--model_config=gpt-4o-2024-05-13 \
--input_path=instructions_with_rubrics.json \
--output_path=completions.json

# Generate evaluations based on provided completions
rubric_eval \
--model_config=gpt-4o-2024-05-13 \
--input_path=completions.json \
--output_path=evaluations.json
```

It should create a `model_card.json` file at your current working directory.
The content of `model_card.json` should look like:
```bash
$ cat model_card.json

[
    {
        "model_name":"gpt-4o-2024-05-13",
        "evaluator":"gpt-4o-2024-05-13_CoT_v0",
        "num_evaluations":5,
        "mean_of_avg_score":3.0,
        "std_of_avg_score":0.3741657387
    }
]
```
`mean_of_avg_score` is the average score earned by the completor model, based on the rubrics in `instructions_with_rubrics.json`.


## Code
- `src/rubric_eval` contains the main code.
  - `annotators.py` contains the code for all annotators that inherit from AlpacaEval (all the steps besides summarizer). AlpacaEval annotators are classes that take a dataframe as input (with columns of potential fields that should be inputed in the prompt to the LLM) and outputs a dataframe with the same columns + the outputs of the LLM. The annotators are the main classes that used in the pipeline.
  - `*_configs` specifies all the hyperparameters and prompt for different potential anntoators (i.e. instances of classes in `annotators.py`). As an example, `rubric_brainstormer_configs/gpt-4o-2024-05-13_CoT_v0/configs.yaml` specifies all the hyperparameters for the rubric_brainstomer we use. Note: I would highly suggest understanding the different hyperparameters, read the [docstrings](https://github.com/tatsu-lab/alpaca_eval/blob/e3993450e2c6d5b5fadd74e0c79fa261c0e98112/src/alpaca_eval/annotators/base.py#L535) and open an issue if something is unclear. The most important are:
```yaml
gpt-4o-2024-05-13_CoT_v0: # should be the name of the directory
  prompt_template: "gpt-4o-2024-05-13_CoT_v0/prompt.txt" # prompt template. Curly braces {column} will be replaced by values of the corresponding column in the dataframe
  fn_completions: "openai_completions" # functions to use to get completions. see https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py
  completions_kwargs: # kwargs to the completions functions. E.g. for openai it's all the openai decoding kwargs
    model_name: "gpt-4o-2024-05-13"
    max_tokens: 4096
    # ...
  fn_completion_parser: "json_parser" # how to parse the completions. typically json parser.
```
