# <a href="https://huggingface.co/spaces/vbhat4/rubriceval" target="_blank"><img src="https://raw.githubusercontent.com/YannDubs/RubricEval/main/docs/rubriceval_icon.png" width="35"></a> [RubricEval](https://huggingface.co/spaces/vbhat4/rubriceval): Scalable Expert Evaluation of Language Models

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

To evaluate model outputs on the RubricEval benchmark, run the following two steps:

1. **Generate model outputs on the RubricEval instructions**: create a file `outputs.json` that is a [JSON file of the RubricEval evaluation set](...) with an additional `outputs` column containing the model outputs to evaluate. You can use you favorite generation pipeline, or our pipeline based on [AlpacaEval](): 

```bash
rubric_eval generate_outputs \
--output_path=outputs.json \
--model_configs=<model_to_evaluate> # e.g. gpt-4o-2024-05-13
```

2. **Evaluate the model outputs**: evaluate the previously generated models outputs using the RubricEval LLM-judge. 


```bash
export OPENAI_API_KEY=<your_api_key> # for more complex configs, e.g. using Azure or switching clients see https://github.com/tatsu-lab/alpaca_eval/tree/main/client_configs/README.md 
rubric_eval --input_path=outputs.json 
```

This will print the leaderboard in the terminal and generate an `evaluation_report.md` file at [...].



Both previous steps can be done in one step by running `
rubric_eval --model_configs=<model_to_evaluate>`

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
   For more efficient rubric creation, we suggest and provide the code for an LLM and expert collaborative three-step pipeline where at each step an LLM generates increasing levels of details for the rubric, which are then reviewed and modified by the expert. Of course, each step could also be done solely by an LLM or an expert. In particular, for each instruction, we suggest the following steps:

   1. **Unstructured expert information** (optional) experts write a high-level unstructured dump of information that could be useful to generate the rubric. E.g. a checklist of things that should be considered for evaluation, the user intent, axes to consider, etc. E.g. below
   
      [add figure]
   
      In our codebase, we expect the unstructured information to be a string in the field `useful_info_to_eval_instruction` for each instruction. 

      In our experience, this is easier to write by looking at the answer of an LLM on that instruction. In our codebase, this can be done using ```rubric_eval generate_outputs --input_path <instructions.json> --output_path <instructions_with_ref.json>```.

   2. **Rubric brainstorming** an LLM converts the unstructured information to a high-level structured rubric. Experts can then review and modify the brainstormed rubric. E.g. below
   
      ![rubric_brainstorming.png](figures%2Frubric_brainstorming.png)

      In our codebase, this can be done using 
   
      ```bash
      rubric_eval brainstorm_rubrics \
        --input_path <instructions.json> \
        --output_path <instructions_with_brainstorms.json>
      ```
      which adds a `brainstormed_rubric` column using [`RubricBrainstormer`]().

   3. **Rubric generation** the LLM generates the detailed rubric given the brainstormed rubric. Experts can then review and modify the generated rubric. E.g. below
   
      ![rubric_generating.png](figures%2Frubric_generating.png)

        In our codebase, this can be done using 
      
      ```bash
      rubric_eval generate_rubrics \
        --input_path <instructions_with_brainstorms.json> \
        --output_path <instructions_with_rubrics.json>
      ```
         
      which adds a `rubric` column using [`RubricGenerator`]().
   
3. **Model outputs** to apply a previously created RubricEval benchmark, you first have to generate the models outputs on the instructions. 

   Our codebase assumes that the outputs are added to the JSON file with the following format:

   ```json
   [
      {"instruction":  "<instruction to evaluate on>", 
       "category": ...,
       "rubric": {...},
       "outputs": "<model outputs>",
       ... 
     },
     ... 
   ]
   ```
   The outputs can be generated using you favorite decoding pipeline. For simplicity, we also provide a simple interface for generating outputs based on [AlpacaEval](): 

   ```bash
   rubric_eval generate_outputs \
    --input_path <instructions_with_rubrics.json> \
    --output_path <outputs.json> \
    --model_configs <model_to_evaluate>
   ```

4. **Evaluation** the next step is to have an LLM evaluate the model's outputs conditioned on the instruction-specific rubric. Evaluation by an LLM but conditioned on an expert-created rubric ensures that the evaluation is trustworthy and interpretable (thanks to the rubric), while being scalable (thanks to the LLM). The evaluation results in both a criteria-based score and interpretable feedback.

   In our codebase, this can be done using 

   ```bash
   rubric_eval evaluate \
    --input_path <outputs.json> \
    --output_path <evaluations.json> \
    --summarize False
   ```
   
   which will add a `score_per_criteria` and `feedback_per_criteria` column to the JSON file using [`Evaluator`]()resulting in the following format:

   ```json
    [
        {"instruction": ..., 
         "category": ...,
         "rubric": {...},
         "outputs": ...,
         "feedback_per_criteria": {
           "<axes 1>": "<feedback>",
           "<axes 2>": "<feedback>",
           ...
         },
         "score_per_criteria": {
            "<axes 1>": <score>,
            "<axes 2>": <score>,
            ...
         },
         ... 
      },
      ... 
    ]
    ```

6. **Evaluation report**  One of the main benefits of the RubricEval pipeline is that the feedback and the rubric can provide finegrained interpretability as that what the model did well/wrong and why it was given a certain score.
That said going through all the rubrics and feedback would be too time-consuming, so we use an LM to summarize the qualities and issues of the outputs from the model being evaluated. The final score and summary are then packaged into an evaluation report for the model.

   In our codebase, this can be done using 

   ```bash
   rubric_eval generate_report \
    --input_path <evaluations.json> \
    --output_path <evaluation_report.md>
   ```

   which will generate a markdown file with the following format:

   ```markdown
   # Evaluation report for <model_name> on <benchmark> 
   
   ## Overview
   
   **Score**: <score> ± <sem>
   
   **Feedback**: 
   
   Pro:
    - <feedback_1>
    - <feedback_2>
   Con:
    - <feedback_1>
    - <feedback_2> 
   
   
   ## Details
   
   **Model**: <model_name>
   **Date**: <date>
   
   **Dataset**: <dataset_name>
   **Num. instructions**: <num_instructions>
   
   **Evaluator**: <evaluator_name>
   **Summarizer**: <evaluator_name>
   **RubricEval version**: <version>
   
   **Evaluation time**: <time>
   **Evaluation cost**: <cost>
   
   ## Quantitative evaluation
   
   **Score**: <score> ± <sem>
   
   **Score per category**:
   
   |    Category  |   Score   |   SEM   |
   |--------------|-----------|---------|
   | <category_1> | <score_1> | <sem_1> |
   
   **Leaderboard**:
   
   |    Model    |   Score   |   SEM   |
   |-------------|-----------|---------|
   | <model_1>   | <score_1> | <sem_1> |
   | <model_2>   | <score_2> | <sem_2> |
    
   
   ## Qualitative evaluation
   
   **Feedback**:
   
   Pro:
    - <feedback_1>
    - <feedback_2>
   Con:
    - <feedback_1>
    - <feedback_2>
   
   **Feedback per category**:
   
    |    Category  |   Feedback   |
    |--------------|--------------|
    | <category_1> | <feedback_1> |
    | <category_2> | <feedback_2> |
   
   ```  


----


## Code
- `src/rubric_eval` contains the main code.
  - `annotators.py` contains the code for all annotators that inherit from AlpacaEval (all the steps besides summarizer). AlpacaEval annotators are classes that take a dataframe as input (with columns of potential fields that should be inputed in the prompt to the LLM) and outputs a dataframe with the same columns + the outputs of the LLM. The annotators are the main classes that used in the pipeline.
  - `*_configs` specifies all the hyperparameters and prompt for different potential anntoators (i.e. instances of classes in `annotators.py`). As an example, `rubric_brainstormer_configs/gpt-4o-2024-08-06_CoT_v0/configs.yaml` specifies all the hyperparameters for the rubric_brainstomer we use. Note: I would highly suggest understanding the different hyperparameters, read the [docstrings](https://github.com/tatsu-lab/alpaca_eval/blob/e3993450e2c6d5b5fadd74e0c79fa261c0e98112/src/alpaca_eval/annotators/base.py#L535) and open an issue if something is unclear. The most important are:
```yaml
gpt-4o-2024-08-06_CoT_v0: # should be the name of the directory
  prompt_template: "gpt-4o-2024-08-06_CoT_v0/prompt.txt" # prompt template. Curly braces {column} will be replaced by values of the corresponding column in the dataframe
  fn_completions: "openai_completions" # functions to use to get completions. see https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py
  completions_kwargs: # kwargs to the completions functions. E.g. for openai it's all the openai decoding kwargs
    model_name: "gpt-4o-2024-05-13"
    max_tokens: 4096
    # ...
  fn_completion_parser: "json_parser" # how to parse the completions. typically json parser.
```
