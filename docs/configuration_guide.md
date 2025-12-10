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
2. [Evaluation Configs](#evaluation-configs)
3. [Assessment Configs](#assessment-configs)
4. [Prompt Configs](#prompt-configs)
5. [Path Resolution](#path-resolution)

---

## Configuration Hierarchy

The evaluation framework uses three levels of configuration:

```
Evaluation Config (Top Level)
├── Specifies: pipeline type, models, assessments, metrics, output directory
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
    "models": ["Model Name 1", "Model Name 2"],
    "assessments": ["assess_example.json"],
    "outdir": "output_directory_name"
}
```

**Fields:**
- `pipeline_type` (required): Must be `"full"`
- `models` (required): Array of model names (must match model's `name` property)
- `assessments` (required): Array of assessment config filenames
- `outdir` (optional): Output directory name (default: `evaluation_results`)

#### Metrics-Only Pipeline

The metrics-only pipeline has two ways to specify inference results:

**Method 1: Direct File List**
```json
{
    "pipeline_type": "metrics_only",
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
        "../inference_engine/prompts/prompt_file_1.json",
        "../inference_engine/prompts/prompt_file_2.json"
    ],
    "metrics": ["MMLU_Accuracy", "ROUGE_Score"]
}
```

**Fields:**
- `name` (required): Identifier for this assessment (used in results grouping)
- `prompts` (required): Array of paths to prompt config files
- `metrics` (required): Array of metric names to calculate

### Path Resolution for Prompts

Prompt file paths in assessment configs are resolved in this order:
1. Absolute path → use as-is
2. Relative to current working directory
3. Relative to `assessment_configs_dir`

**Recommendation:** Use relative paths from the assessment_configs directory:
```json
"prompts": ["../inference_engine/prompts/prompt_mmlu_test.json"]
```

### Example

```json
{
    "name": "mmlu_global_facts_logical_fallacies",
    "prompts": [
        "../inference_engine/prompts/prompt_mmlu_logical_fallacies_test_few_shot.json",
        "../inference_engine/prompts/prompt_mmlu_global_facts_test_few_shot.json"
    ],
    "metrics": ["MMLU_Accuracy"]
}
```

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
- `style` (required): Prompt style - one of `"basic"`, `"single_token"`, `"multi_token"`
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

3. **Prompt configs**: Use relative paths from assessment config location
   ```json
   "prompts": ["../inference_engine/prompts/my_prompts.json"]
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
