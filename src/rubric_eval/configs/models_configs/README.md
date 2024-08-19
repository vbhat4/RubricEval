# Model Configs

This directory contains the configuration files for models that can be used for generating completions in RubricEval using 

```bash
rubric_eval generate_outputs \
    --input_path <instructions_with_rubrics.json> \
    --output_path <instructions_with_completions.json> \
    --model_configs <model_configs_dir>
    
rubric_eval evaluate \
    --input_path <instructions_with_completions.json> \
    --output_path <evaluations.json> 
```

or in a single command `rubric_eval generate_outputs_and_evaluate --input_path <instructions_with_rubrics.json> --output_path <evaluations.json> --model_configs <model_configs_dir>`.

Note that you can generate completions from any method/model in this directory, as well as any of the 100+ methods in alpaca_eval's [models_configs directory](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/models_configs) using the same command.