# License

ELM Code Library

Copyright 2025 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE
MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO
WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR
MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL.
CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT
TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT-style license, please see license.txt or contact
permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release
and unlimited distribution.  Please see Copyright notice for non-US Government
use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to
its own license.

DM25-1265

# Getting Started

This document will walk through how to run your first evaluation from installation. You will learn how to:
1. Install the `elm` package
2. Set up your environment for local inference using a `llama3.2-1B` model 
3. Create the prompt, assessment, and evaluation configuration files for your pipeline
4. Running the pipeline
5. Reviewing the evaluation results

## Installation

After cloning the repository, you will need to install the `elm` package.

Navigate to the root directory of the repository and run the package setup:
```bash
pip install -e .
```
This will trigger `setup.py` to install the package and its requirements.

## Environment Setup and Downloading Model Weights

For this example, we will need the [Llama-3.2-1B model files from HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-1B).

You will need a HuggingFace account with access to this model and be logged into your HuggingFace account in your terminal session.

Download the model files to a directory of your choice. Make note of this directory as we will be referencing it later. Here is an example command that downloads the model files to `~models/llama`
```bash
hf download meta-llama/Llama-3.2-1B --local-dir ~/models/llama
```

Create an environment configuration file in `llm-evaluation/elm/inference_engine/environment_configs/your_env_config.json` to specify the directory containing the newly downloaded model files.
```json
{
    "name": "getting_started_environment",
    "models": [
        {
            "model_name": "LLaMa 3.2 1B",
            "model_family": "Llama",
            "weights_dir": "~/models/llama",
            "tokenizer_dir": "~/models/llama",
            "cache_dir": "~/models/llama"
        }
    ]
}
```

With the model files downloaded and the corresponding environment configuration file updated with the proper directory locations, we can now proceed to setting up our evaluation pipeline.

## Creating the Evaluation Pipeline

To create an evaluation pipeline, we will need a prompt file, an assessment configuration file, and an evaluation configuration file. Let's get those setup.

### Creating a prompt file

Prompt files contain the text prompts which will be sent to the language model(s), along with an optional ground truth field.

By convention, prompt files should be located in `elm/inference_engine/prompts/`.

A prompt file should adhere to the following schema:
```json
[
    {
        "name": "unique_prompt_identifier",
        "style": "basic",
        "text": "The prompt text sent to the model...",
        "gt_text": "Expected answer"
    }
]
```
**Fields:**
- `name` (required): Unique identifier for this prompt
- `style` (required): Prompt style. `basic` is the only style currently supported.
- `text` (required): The actual prompt text
- `gt_text` (optional): Ground truth text for comparison metrics (e.g., ROUGE, accuracy)

**Note:** Prompt config files are JSON arrays containing multiple prompt objects.

For our first pipeline, let's create a file `prompt_getting_started.json` located at `elm/inference_engine/prompts/prompt_getting_started.json`.
Let's add some multi-choice question and answer style prompts to our new prompt file:
```json
[
    {
        "name": "mmlu_style_question_1",
        "style": "basic",
        "text": "What is the capital of France?\nA. London\nB. Berlin\nC. Paris\nD. Madrid\nAnswer:",
        "gt_text": "C"
    },
    {
        "name": "mmlu_style_question_2",
        "style": "basic",
        "text": "Which planet is closest to the sun?\nA. Venus\nB. Mercury\nC. Earth\nD. Mars\nAnswer:",
        "gt_text": "B"
    }
]
```

Now that we have a prompt file with some prompts, let's make an assessment configuration file to reference it.


### Creating an assessment configuration file

Assessment configs define a set of prompts and the metrics to calculate on their results.

**Location:** 
`elm/evaluation_engine/assessment_configs/`

**Schema:**
```json
{
    "name": "assessment_identifier",
    "prompts": [
        "prompt_file_1.json",
        "prompt_file_2.json"
    ],
    "metrics": ["MMLU_Accuracy", "ROUGE_Score"]
}
```

**Fields:**
- `name` (required): Identifier for this assessment (used to group results)
- `prompts` (required): Array of prompt config files
- `metrics` (required): Array of metric names to calculate

**Path Resolution for Prompts:**
The engine will look for prompt files in the default directory of `elm/inference_engine/prompts/`.

For our first pipeline, let's create a file `assess_getting_started.json` located at `elm/evaluation_engine/assessment_configs/assess_getting_started.json`.
This config will reference the prompt file we created in the previous step, and the MMLU accuracy metric which will score the model responses based on the ground truth labels provided in each corresponding prompt.
```json
{
    "name": "assess_getting_started",
    "prompts": [
        "prompt_getting_started.json"
    ],
    "metrics": ["MMLU_Accuracy"]
}
```

### Creating an evaluation configuration file

Evaluation configs are the entry point for running evaluations. They define what models to use, which assessments to run, and what type of pipeline to execute.

### Location
- Default directory: `elm/evaluation_engine/evaluation_configs/`

**Schema**
```json
{
    "outdir": "output_directory_name",
    "pipeline_type": "full",
    "environment_config": "env_config.json",
    "models": [
        {"name": "Model Name 1"},
        {"name": "Model Name 2"}
    ],
    "assessments": [
        {"config": "assess_example.json"}
    ]
}
```

**Fields:**
- `pipeline_type` (required): Must be `"full"`
- `models` (required): Array of model names (must match model's `name` property)
- `assessments` (required): Array of assessment config filenames
- `outdir` (required): Output directory name (results will be saved to `elm/evaluation_engine/evaluation_results/outdir`)

Evaluation configs can also specify a `metrics_only` pipeline for calculating metrics using pre-existing inference results. Details for setting up a `metrics_only` pipeline can be found in `docs/configuration_guide.md`. For now, we'll focus on running a `full` pipeline for our first evaluation.

Create a file `eval_getting_started.json` located at `elm/evaluation_engine/evaluation_configs/eval_getting_started.json`.

This file will reference the environment config, assessment config and model we setup in previous steps:
```json
{
    "outdir": "my_first_evaluation",
    "pipeline_type": "full",
    "environment_config": "env_getting_started.json",
    "models": [
        {"name": "LLaMa 3.2 1B"}
    ],
    "assessments": [
        {"config": "assess_getting_started.json"}
    ]
}
```

## Running the Evaluation Pipeline

Now that the pipeline is setup, we can navigate to the `elm/evaluation_engine/` directory and execute the pipeline with:
```bash
python EvaluationEngine.py -c evaluation_configs/eval_getting_started.json
```

The pipeline will execute, saving the individual inference results and the final evaluation report to the `outdir` specified by the evaluation config file.

## Reviewing Evaluation Results

As our first pipeline executes, you should see terminal output similar to the following:
```
Evaluation Engine initialized at 2025-11-06 19:38:26.732788
Available metrics: ['MMLU_Accuracy', 'ROUGE_Score']
Available models: ['LLaMa 3.2 1B']
Loaded evaluation config: /your-workspace/llm-evaluation/elm/evaluation_engine/evaluation_configs/eval_getting_started.json
Executing full pipeline...
Starting full evaluation pipeline...

Processing model: LLaMa 3.2 1B
Loaded assessment config: /your-workspace/llm-evaluation/elm/evaluation_engine/assessment_configs/assess_getting_started.json
Executing inference for LLaMa 3.2 1B on assessment assess_getting_started
Loading prompts from: prompt_getting_started.json
----------
Loading LLaMa 3.2 1B...
----------
Executing inference: 'LLaMa 3.2 1B' with 'mmlu_style_question_1'...
Results file saved: /your-workspace/llm-evaluation/elm/evaluation_engine/evaluation_results/my_first_evaluation/run_eval_20251106_193830/LLaMa 3.2 1B/assess_getting_started/inference_results/LLaMa_3.2_1B_20251106_193904.json
----------
Executing inference: 'LLaMa 3.2 1B' with 'mmlu_style_question_2'...
Results file saved: /your-workspace/llm-evaluation/elm/evaluation_engine/evaluation_results/my_first_evaluation/run_eval_20251106_193830/LLaMa 3.2 1B/assess_getting_started/inference_results/LLaMa_3.2_1B_20251106_193906.json
Inference complete. Generated 2 result files.
Calculating metrics for assessment: assess_getting_started
Calculating MMLU_Accuracy...
MMLU_Accuracy: completed
Evaluation report saved: /your-workspace/llm-evaluation/elm/evaluation_engine/evaluation_results/my_first_evaluation/run_eval_20251106_193830/evaluation_report_eval_20251106_193830.json
Evaluation finished after 36.23 seconds
Evaluation complete! Report saved to: /your-workspace/llm-evaluation/elm/evaluation_engine/evaluation_results/my_first_evaluation/run_eval_20251106_193830/evaluation_report_eval_20251106_193830.json
```

The output details where to find the inference results files and the evaluation report generated by the pipeline. The evaluation report should look something like this:
```
{
    "evaluation_metadata": {
        "run_id": "eval_20251106_193830",
        "evaluation_config": "evaluation_configs/eval_getting_started.json",
        "timestamp": "2025-11-06T19:38:30.897899",
        "pipeline_type": "full",
        "total_models": 1,
        "total_assessments": 1,
        "total_execution_time": 36.22903880701051
    },
    "model_results": [
        {
            "model_name": "LLaMa 3.2 1B",
            "assessments": [
                {
                    "name": "assess_getting_started",
                    "config": "/your-workspace/llm-evaluation/elm/evaluation_engine/assessment_configs/assess_getting_started.json",
                    "execution_time": 36.22689175000414,
                    "total_prompts": 2,
                    "metric_summaries": {
                        "MMLU_Accuracy": {
                            "counts": {
                                "total_items": 2,
                                "scored_items": 2,
                                "skipped_items": 0,
                                "failed_items": 0,
                                "correct_answers": 1,
                                "incorrect_answers": 1
                            },
                            "scores": {
                                "accuracy": 0.5,
                                "accuracy_percentage": 50.0
                            },
                            "issues": []
                        }
                    },
                    "prompt_results": [
                        {
                            "name": "mmlu_style_question_1",
                            "model_output": " C\nExplanation:   Paris is the capital of France.",
                            "inference_time": 1.6795748290023766,
                            "source_file": "/your-workspace/llm-evaluation/elm/evaluation_engine/evaluation_results/my_first_evaluation/run_eval_20251106_193830/LLaMa 3.2 1B/assess_getting_started/inference_results/LLaMa_3.2_1B_20251106_193904.json",
                            "gt_text": "C",
                            "metric_details": {
                                "MMLU_Accuracy": {
                                    "status": "ok",
                                    "errors": [],
                                    "correct": true
                                }
                            }
                        },
                        {
                            "name": "mmlu_style_question_2",
                            "model_output": " A\nExplanation: The planets move around the sun in orbits. The orbits are shaped by the gravity of the sun. The orbits of the planets from inside to outside are Mercury, Venus, Earth, Mars.",
                            "inference_time": 1.521869465999771,
                            "source_file": "/your-workspace/llm-evaluation/elm/evaluation_engine/evaluation_results/my_first_evaluation/run_eval_20251106_193830/LLaMa 3.2 1B/assess_getting_started/inference_results/LLaMa_3.2_1B_20251106_193906.json",
                            "gt_text": "B",
                            "metric_details": {
                                "MMLU_Accuracy": {
                                    "status": "ok",
                                    "errors": [],
                                    "correct": false
                                }
                            }
                        }
                    ]
                }
            ]
        }
    ]
}
```