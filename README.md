# RubricEval

---

# Quick Start

To install the stable release, run

```bash
pip install rubric-eval
```

To install the nightly version, run

```bash
pip install git+https://github.com/tatsu-lab/rubric_eval
```

Then you can use it as follows:

```bash
export OPENAI_API_KEY=<your_api_key> # for more complex configs, e.g. using Azure or switching clients see https://github.com/tatsu-lab/alpaca_eval/tree/main/client_configs/README.md 
python scripts/create_example_instructions.py instructions.json  # This will generate an example instructions.json
rubric_eval get_rubrics --input_path=instructions.json --output_path=instructions_with_rubrics.json
rubric_eval get_completions --model_config=gpt-4o-2024-05-13 --input_path=instructions_with_rubrics.json --output_path=completions.json
rubric_eval --model_config=gpt-4o-2024-05-13 --input_path=completions.json --output_path=evaluations.json
```

It should create a `model_card.json` file at your current working directory (along with other files such as `evaluations.json`).
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


## Pipeline
The pipeline for the project is as follows:
1. **Writing the instructions** this is to collect the instructions that we'll input into the model as well as potential expert-written additional information that could improve the quality of the next rubric generation (e.g. a checklist per instruction). 
   - **Real-world** in the real world, experts would write a "small" number of instructions (e.g. 200) and associated bullet points that they think are important to consider when evaluating the output on each instruction. For experts this will be the most costly part of the pipeline, but it's only done once so it should be worth it. Ideal instructions for RubricEval are from some expert domain such that:
     - (1) models can't evaluate on their own (but can apply the rubric)
     - (2) experts are expensive and can thus not do all the evaluations on their own
     - (3) the final use case is high-stakes and thus require highly trusted benchmark and interpretability in the results. E.g. of domains that are well suited: law, medical, finance, etc. 
2. **Brainstorming the rubrics** given the instruction and additional information, an LLM or an expert could write some high-level rubric as seen below.
![rubric_brainstorming.png](figures%2Frubric_brainstorming.png)
    - **Real-world** in the real-world the additional cost of having experts write the high-level rubrics is minimal compared to adding a checklist (the only difference is that it has to be structured by different axes. Our belief is that having expert-written high-level rubrics would give much more realistic final rubrics and thus be worth the cost.
    - **Benchmark** for the benchmark, we provide a `RubricBrainstormer` that aims to convert the usntructured additional_information (e.g. checklist) to the high-level rubric. This is to allow converting any dataset to our format and should be used for example with WildBench.
3. **Generating the rubrics** given the high-level rubric, we can generate the final rubric. This is done by the `RubricGenerator` which is a simple LLM that takes in the high-level rubric and the instruction and generates the final rubric. This step should be simple for LLMs but tedious for humans. I would thus suggest always having an LLM do this step, with optional human review & modification. 
![rubric_generating.png](figures%2Frubric_generating.png)
4. **Decoding/Inference of the models** there's nothing new here, we just need to predict what different models would say. This is done by the `Completor` and `alpaca_eval` essentially already provides the code to evaluate hundreds of models.
5. **Evaluation** the `Evaluator` then takes in the predictions and the rubric and outputs some feedback, the selected category in the rubric, and the final score. This should be done only by LLM for scalability. This will likely be the hardest for the model in some domains (e.g. if it's bad at math it will be hard for it to evaluate math instructions). But for many valuable domains (healthcare, legal, ...)  I'm confident that current models can do a good job.
6. **Summarizer**  One of the main benefits of RubricEval is that it can provide more finegrained interpretability into what the model does well/wrong and why it was given a certain score. THe problem is that no one will read N>=200 feedback and selected rubrics. We would thus need a summarizer that takes in the feedback and selected rubric and generates a summary / model card of what the model does well and badly. This is a bit different from the other steps in that the function is many-to-one (many feedbacks to one summary), which is why we don't have a minimal class that inherits from AlpacaEval to do that. A minimal version of the summarizer is `summarize_results` but this should really be rewritten. Note that providing all the outputs & instructions & feedback & rubrics will probably exceed the context length of the llm judge. So I would suggest having a two (or more) layer summarizier pipeline, which recursively applies summarizers to the output of the previous summarizer. Note that the summarizier will need to be prompt tuned to ensure that it provides detailed and actionable summaries of the model's performance that are self contained (i.e. the human shouldn't need to know about the instructions to understand the summary).

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
