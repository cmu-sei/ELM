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

# Configuration Guide

Complete reference for all configuration files in the evaluation framework.

## Table of Contents

1. [Configuration Hierarchy](#configuration-hierarchy)
2. [Environment Configs](#environment-configs)
3. [Evaluation Configs](#evaluation-configs)
4. [Assessment Configs](#assessment-configs)
5. [Prompt Configs](#prompt-configs)
6. [Path Resolution](#path-resolution)

---

## Configuration Hierarchy

The evaluation framework uses the following configuration files:

```
Environment Config
├── Defines: Available models and paths to local weights
│
Evaluation Config (Top Level)
├── Specifies: pipeline type, models, assessments, metrics, output directory, hyperparameter overrides (optional), quantization config overrides (optional)
├── References: Assessment configs (for full pipeline) or inference results (for metrics_only)
│
└─> Assessment Config (Mid Level)
    ├── Specifies: assessment name, prompts, metrics
    ├── References: Prompt config files
    │
    └─> Prompt Config (Bottom Level)
        └── Specifies: individual prompts with optional ground truth
```

**Key principle:** Each level references the next level by filename, and the engine resolves paths automatically.

---

## Environment Configs

Environment configs define available models and their file system locations. Required for both inference and evaluation engines.

### Location
- Default directory: `elm/inference_engine/environment_configs/`
- Specified in inference/evaluation configs via `environment_config` field

### Schema
```json
{
    "name": "test_environment",
    "models": [
        {
            "model_name": "LLaMa 3.2 1B",
            "model_family": "Llama",
            "weights_dir": "/path/to/weights_dir",
            "tokenizer_dir": "/path/to/tokenizer_dir",
            "cache_dir": "/path/to/cache_dir"
        },
        {
            "model_name": "OpenAI o4 Mini",
            "model_family": "OpenAI",
            "model_code": "o4-mini"
        }
    ]
}
```

**Required Fields:**
- `name`: Environment identifier
- `models`: Array of model specifications
  - `model_name`: Unique name referenced in configs
  - `model_family`: Model class name (must match file in `languagemodels/`)
  - `model_code`: Unique identifier for API-based models

---

## Evaluation Configs

Evaluation configs are the entry point for running evaluations. They define what models to use, which assessments to run, and what type of pipeline to execute.

### Location
- Default directory: `elm/evaluation_engine/evaluation_configs/`
- Can be overridden with `--evaluation-configs-dir` CLI argument

### Schema

#### Full Pipeline

```json
{
    "pipeline_type": "full",
    "hyperparameters": {
        "temperature": 0.7,
        "max_new_tokens": 256,
        "top_p": 0.9
    },
    "models": [
        {
            "name": "Example Model",
            "hyperparameters": {
                "temperature": 0.9
            }
        }
    ],
    "assessments": [
        {
            "config": "example_assessment.json",
            "hyperparameters": {
                "max_new_tokens": 512
            }
        }
    ],
    "outdir": "output_directory_name"
}
```

**Fields:**
- `pipeline_type` (required): Must be `"full"`
- `environment_config` (required): Path to environment config file
- `hyperparameters` (optional): Global-level generation parameters (lowest priority)
- `models` (required): Array of model specifications
  - `name` (required): Model name from environment config
  - `hyperparameters` (optional): Model-level overrides (highest priority)
  - `quantization_config` (optional): Quantization settings for model loading
- `assessments` (required): Array of assessment specifications
  - `config` (required): Assessment config filename
  - `hyperparameters` (optional): Assessment-level overrides (middle priority)
- `outdir` (optional): Output directory name (default: `evaluation_results`)

**Hyperparameter Priority:** Model > Assessment > Global

#### Metrics-Only Pipeline

The metrics-only pipeline has two ways to specify inference results:

**Method 1: Direct File List**
```json
{
    "pipeline_type": "metrics_only",
    "environment_config": "test_env.json",
    "metrics": ["MMLU_Accuracy", "ROUGE_Score"],
    "inference_results": [
        "/absolute/path/to/inference_result_1.json",
        "/absolute/path/to/inference_result_2.json"
    ],
    "outdir": "metrics_only_output"
}
```

**Method 2: From Evaluation Report**
```json
{
    "pipeline_type": "metrics_only",
    "environment_config": "test_env.json",
    "metrics": ["MMLU_Accuracy"],
    "inference_results": {
        "type": "from_evaluation_report",
        "report_path": "/path/to/evaluation_report_eval_20250904_185222.json",
        "filter": {
            "models": ["LLaMa 3.2 1B"],
            "assessments": ["mmlu_example"]
        }
    },
    "outdir": "reanalysis_output"
}
```

**Fields:**
- `pipeline_type` (required): Must be `"metrics_only"`
- `environment_config` (required): Path to environment config file
- `metrics` (required): Array of metric names to calculate
- `inference_results` (required): Either:
  - Array of file paths (absolute paths to inference result JSON files)
  - Object with `type: "from_evaluation_report"` and:
    - `report_path` (required): Path to evaluation report JSON file
    - `filter` (optional): Object with `models` and/or `assessments` arrays to filter results
- `outdir` (optional): Output directory name

### Validation

The evaluation engine validates configs using Pydantic models. Common errors:

- Missing required fields for pipeline type
- Invalid model names (not in `available_models`)
- Assessment config files not found
- Inference result files not found (for metrics_only)
- Invalid JSON format

---

## Assessment Configs

Assessment configs define a set of prompts and the metrics to calculate on their results. They are used only in the **full pipeline**.

### Location
- Default directory: `elm/evaluation_engine/assessment_configs/`
- Can be overridden with `--assessment-configs-dir` CLI argument

### Schema

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
- `name` (required): Identifier for this assessment (used in results grouping)
- `prompts` (required): Array of prompt config filenames
- `metrics` (required): Array of metric names to calculate


### Example

```json
{
    "name": "mmlu_global_facts_logical_fallacies",
    "prompts": [
        "prompt_mmlu_logical_fallacies_test_few_shot.json",
        "prompt_mmlu_global_facts_test_few_shot.json"
    ],
    "metrics": ["MMLU_Accuracy"]
}
```

---

## Inference Configs

Inference configs are used by the Inference Engine for standalone inference runs (outside of evaluation pipelines).

### Location
- Default directory: `elm/inference_engine/inference_configs/`

### Schema
```json
[
    {
        "output_directory": "results/experiment_1",
        "environment_config": "test_env.json",
        "hyperparameters": {
            "temperature": 0.7,
            "max_new_tokens": 256,
            "top_p": 0.9
        },
        "inference_sets": [
            {
                "models": [
                    {
                        "name": "LLaMa 3.2 1B",
                        "hyperparameters": {
                            "temperature": 0.9
                        },
                        "quantization_config": {
                            "load_in_8bit": true
                        }
                    },
                    {
                        "name": "LLaMa 3.1 8B Instruct"
                    }
                ],
                "prompts": ["test_prompts.json"],
                "hyperparameters": {
                    "top_p": 0.95
                }
            }
        ]
    }
]
```

**Fields:**
- `output_directory` (required): Directory for inference results
- `environment_config` (required): Path to environment config file
- `hyperparameters` (optional): Global-level generation parameters
- `inference_sets` (required): Array of model/prompt combinations
  - `models` (required): Array of model specifications
    - `name` (required): Model name from environment config
    - `hyperparameters` (optional): Model-level overrides
    - `quantization_config` (optional): Quantization settings
  - `prompts` (required): Array of prompt config filenames
  - `hyperparameters` (optional): Set-level overrides

**Hyperparameter Priority:** Model > Set > Global

---

## Prompt Configs

Prompt configs contain the actual text prompts sent to language models, along with optional ground truth for metric calculation.

### Location
- Typically: `elm/inference_engine/prompts/`
- Can be anywhere (referenced by assessment configs)

### Schema

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
- `style` (required): Prompt style. Current support only for `"basic"`
- `text` (required): The actual prompt text (minimum 1 character)
- `gt_text` (optional): Ground truth text for comparison metrics (e.g., ROUGE, accuracy)

**Note:** Prompt config files are JSON arrays containing multiple prompt objects.

### Ground Truth

Metrics like MMLU_Accuracy and ROUGE_Score require ground truth for comparison:
- `gt_text` is validated to be non-empty if provided
- Metrics will skip prompts without required ground truth
- Skipped prompts are reported in metric summary `issues` array

### Example

```json
[
    {
        "name": "mmlu_question_1",
        "style": "basic",
        "text": "What is the capital of France?\nA. London\nB. Berlin\nC. Paris\nD. Madrid\nAnswer:",
        "gt_text": "C",
        "gt_file": null
    },
    {
        "name": "mmlu_question_2",
        "style": "basic",
        "text": "Which planet is closest to the sun?\nA. Venus\nB. Mercury\nC. Earth\nD. Mars\nAnswer:",
        "gt_text": "B",
        "gt_file": null
    }
]
```

---

## Path Resolution

The evaluation engine uses flexible path resolution to find configuration files.

### Resolution Order

For all config files (evaluation, assessment, prompt):

1. **Absolute paths** → Used verbatim
2. **Relative to current working directory** → Checked first
3. **Relative to appropriate config directory** → Checked second
   - If path starts with the base directory name, that segment is stripped and the rest is resolved

### Examples

Given:
- CWD: `/workspace/elm/evaluation_engine`
- `evaluation_configs_dir`: `/workspace/elm/evaluation_engine/evaluation_configs`
- Filename: `example_config.json`

Resolution attempts:
1. `/workspace/elm/evaluation_engine/example_config.json` (CWD-relative)
2. `/workspace/elm/evaluation_engine/evaluation_configs/example_config.json` (base dir)

Given filename: `evaluation_configs/example_config.json`
Resolution attempts:
1. `/workspace/elm/evaluation_engine/evaluation_configs/example_config.json` (CWD-relative)
2. `/workspace/elm/evaluation_engine/evaluation_configs/example_config.json` (stripped redundant prefix)

### Environment Variables

All paths support environment variable expansion:
```json
{
    "report_path": "$HOME/results/evaluation_report_xxx.json",
    "prompts": ["$PROJECT_ROOT/prompts/my_prompts.json"]
}
```

Use `~` for home directory:
```json
{
    "report_path": "~/evaluation_results/report.json"
}
```

### Best Practices

1. **Evaluation configs**: Use simple filenames, let engine find them in `evaluation_configs/`
   ```bash
   python EvaluationEngine.py -c my_eval.json
   ```

2. **Assessment configs**: Use simple filenames in evaluation config
   ```json
   "assessments": ["assess_mmlu.json"]
   ```

3. **Prompt configs**: Use simple filenames in assessment config
   ```json
   "prompts": ["my_prompts.json"]
   ```

4. **Inference results** (metrics-only): Use absolute paths or environment variables
   ```json
   "inference_results": ["$RESULTS_DIR/run_xxx/inference_results/file.json"]
   ```

---

## Troubleshooting

### "Config file not found"
- Check that filename is correct
- Verify file exists in `evaluation_configs/` or provide absolute path
- Check for typos in filename

### "Invalid evaluation config: Missing required field"
- Verify all required fields for your pipeline type are present
- Check field names match exactly (case-sensitive)
- Ensure arrays and objects are formatted correctly

### "Model 'X' not available"
- Check model name matches exactly (including spaces, capitalization)
- List available models: `engine.available_models.keys()`
- Verify model is registered in `languagemodels/__init__.py`

### "Assessment config not found"
- Check assessment filename in evaluation config
- Verify file exists in `assessment_configs/`
- Check path resolution (see Path Resolution section)

### "Inference result file not found"
- Verify all file paths in metrics_only config exist
- Use absolute paths or check relative path resolution
- For evaluation report: verify report path is correct

### "Metric 'X' not available"
- Check metric name matches exactly
- List available metrics: `engine.available_metrics.keys()`
- Verify metric is registered in `metrics/__init__.py`
