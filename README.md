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

# ELM - Evaluating Language Models

A modular framework for evaluating large language models with configurable prompts, assessments, and metrics.

## Overview

The Evaluation Engine orchestrates LLM inference and metric calculation through a flexible configuration system. Create custom prompts, define assessments, implement new metrics, and run evaluations in two modes: **full** (inference + metrics) or **metrics-only** (metrics on existing results).

### Key Features

- **Custom Prompts**: Define any prompt with optional ground truth
- **Custom Assessments**: Group prompts and specify which metrics to calculate
- **Custom Metrics**: Implement new evaluation criteria via plugin system
- **Model Flexibility**: Add new models through standard interface
- **Structured Output**: Consistent report format with aggregate and per-prompt results

## Quick start

This section briefly overviews installing the package, configuring your environment and model paths, and running an example evaluation.

For a more comprehensive getting started guide, see `docs/getting_started.md`

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
```

2. Install dependencies:

```bash
cd llm-evaluation/
pip install -e .
```

3. Ensure model weights are available at the specified paths in your environment configuration files.

### Interactive Mode

```bash
python Inference_Engine.py
```

The system will prompt you to select from available models and prompts.

### Configuration File Mode

```bash
python Inference_Engine.py -c inference_configs/<example_config_file>.json
```

## Configuration Files

### Prompt Configuration

Create JSON files containing prompt configurations:

```json
[
    {
        "name": "Test Prompt 1",
        "style": "basic",
        "text": "Finish the following sentence: Four score and"
    },
    {
        "name": "Creative Writing",
        "style": "basic", 
        "text": "Write a short story about a robot learning emotions."
    }
]
```

**Prompt Styles:**

- `basic`: Standard prompt processing
- `single_token`: To Be Implemented
- `multi_token`: To Be Implemented

### Inference Configuration

Create comprehensive inference pipelines with JSON configuration files:

```json
[
    {
        "output_directory": "results/experiment_1",
        "environment_config" : "example_env_config.json",
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
                "prompts": [
                    "creative_prompts.json",
                    "reasoning_prompts.json"
                ],
                "hyperparameters": {
                    "top_p": 0.95
                }
            }
        ]
    }
]
```

**Configuration Fields:**

- `output_directory`: Directory for inference results and outputs
- `environment_config`: Path to environment configuration file
- `hyperparameters` (optional): Global-level generation parameters
- `inference_sets`: List of model/prompt combinations
    - `models`: List of model specifications
        - `name`: Model name from environment config
        - `hyperparameters` (optional): Model-level overrides
        - `quantization_config` (optional): Quantization settings
    - `prompts`: List of prompt configuration files
    - `hyperparameters` (optional): Inference set level overrides

**Hyperparameter Priority**: Model > Inference Set > Global (more specific overrides broader)

### Environment Configuration

The system requires an environment configuration to set global variables such as model locations.
Create JSON files containing environment configurations:

```json
{
    "name": "example_config_file",
    "models": 
        [ 
            {
                "model_name": "Example1",
                "model_family": "Examples",
                "weights_dir": "/path/to/model/directory1",
                "tokenizer_dir": "/path/to/model/directory1"
            },
            {
                "model_name": "Example2",
                "model_family": "Examples",
                "weights_dir": "/path/to/model/directory2",
                "tokenizer_dir": "/path/to/model/directory2"
            },
        ]
}
```
All models require a model_name and model_family value. Other required inputs vary by model family and are listed for currently implemented model families. 

**Additional Required Model Inputs:**

- `Llama`: weights_dir, tokenizer_dir, cache_dir
- `T5`: weights_dir, tokenizer_dir
- `OpenAI`: model_code

## Available Model Families

The system currently includes:

- **LLaMa**
- **T5**
- **OpenAI**
  - To use this model you *must* set an environment variable called `OPENAI_API_KEY` to whatever API key you want to use.  For example: `export OPENAI_API_KEY=<API_KEY>`

### Adding New Model Families

1. Create a new model file in `languagemodels/` (e.g., `MyNewModel.py`)
2. Inherit from the `LanguageModel` abstract base class
3. Implement required methods: `name`, `load`, `ask`, `delete`, `log`, `prompter`
4. Add the model name to `__all__` in `languagemodels/__init__.py`
5. Update the import statement in the `__init__.py` file

Example model implementation:

```python
from .LanguageModel import LanguageModel

class Model(LanguageModel):
    def __init__(self, specs):
        self._name = specs["model_name"]
        self.attribute = specs["model_attribute"]
        self.quantization_config_used = None
        # Initialize model-specific parameters
    
    @property
    def name(self):
        return self._name
    
    def load(self, quantization_config=None):
        # Load model into memory, use overrides that were passed in if supported by the model family
        pass
    
    def ask(self, prompt, history=None, hyperparameters=None):
        # Set default hyperparameters and update with overrides that were passed in, if any
        # HuggingFace Transformers based models can use GenerationConfig, see Llama.py for an example
        # Generate response to prompt
        pass
    
    def delete(self):
        # Clean up model from memory
        pass
    
    def log(self):
        # Model-specific logging
        pass
    
    def prompter(self):
        # Handle prompt formatting
        pass
```

## Hardware Monitoring

The inference engine automatically tracks:

- **CPU Usage**: Average and peak utilization during model loading and inference
- **RAM Usage**: Memory consumption in GB and percentage
- **GPU Metrics**: VRAM usage and utilization for each available GPU
- **Timing**: Precise load times and inference times for performance analysis

All metrics are collected at configurable intervals (default: 1 second) and included in result files.

## Output Format

Results are saved as JSON files with comprehensive metadata:

```json
{
    "start_time": "2025-05-28_14:30:25",
    "model_name": "LLaMa 3.2 1B",
    "prompt_config": {
        "name": "Test Prompt 1",
        "style": "basic",
        "text": "Finish the following sentence: Four score and"
    },
    "generation_config": {
        "temperature": 0.9,
        "max_new_tokens": 256,
        "top_p": 0.95,
        "top_k": 50,
        "do_sample": true
    },
    "quantization_config": {
        "load_in_8bit": true,
        "llm_int8_threshold": 6.0
    },
    "load_time": 15.67,
    "inference_time": 2.34,
    "inference_results": "Four score and seven years ago our fathers brought forth...",
    "num_previous_prompts": 0,
    "hardware_metrics": {
        "load": {
            "cpu": {"avg_percent": 45.2, "max_percent": 78.9},
            "ram": {"avg_used_gb": 8.5, "max_used_gb": 12.3},
            "gpu": [...]
        },
        "inference": {
            "cpu": {"avg_percent": 23.1, "max_percent": 41.2},
            "ram": {"avg_used_gb": 12.1, "max_used_gb": 12.8},
            "gpu": [...]
        }
    }
}
```

## Logging

The system maintains detailed logs in `./logs/YYYY-MM-DD.txt` with:

- System startup and shutdown events
- Model loading and unloading operations
- Inference execution tracking
- Hardware metrics collection
- Error reporting and debugging information

## Command Line Options

```bash
python Inference_Engine.py [OPTIONS]

Options:
  -c, --config FILE    JSON configuration file for automated execution
  -h, --help          Show help message and exit
```

---

# Evaluation Engine

A config-driven evaluation framework for assessing language model performance with support for custom benchmarks and metrics.

## Quick Start

```bash
python EvaluationEngine.py -c evaluation_configs/example_eval_config.json
```

## Configuration Files

### Evaluation Configuration

```json
{
    "pipeline_type": "full",
    "outdir": "evaluation_results",
    "environment_config": "example_env.json",
    "hyperparameters": {
        "temperature": 0.7,
        "max_new_tokens": 256
    },
    "models": [
        {
            "name": "LLaMa 3.2 1B",
            "hyperparameters": {
                "temperature": 0.1
            }
        },
        {
            "name": "LLaMa 3.1 8B Instruct"
        }
    ],
    "assessments": [
        {
            "config": "assess_mmlu.json",
            "hyperparameters": {
                "max_new_tokens": 512
            }
        }
    ],
    "metrics": []
}
```

**Pipeline Types:**

- `full`: Run inference and calculate metrics
- `metrics_only`: Calculate metrics from existing inference results

**Configuration Fields:**

- `pipeline_type`: Pipeline mode
- `environment_config`: Path to environment configuration
- `hyperparameters` (optional): Global-level parameters
- `models`: List of models to evaluate
- `assessments`: List of assessment configurations
- `metrics`: Metrics to calculate (for metrics_only pipeline)

### Assessment Configuration

```json
{
    "name": "MMLU Assessment",
    "description": "Massive Multitask Language Understanding benchmark",
    "version": "1.0",
    "prompts": [
        "mmlu_philosophy.json",
        "mmlu_mathematics.json"
    ],
    "metrics": [
        "MMLU_Accuracy"
    ]
}
```

## Available Metrics

- `MMLU_Accuracy`: Accuracy for MMLU-style multiple choice questions
- `ROUGE_Score`: ROUGE scores for summarization tasks
- Additional metrics can be added in `evaluation_engine/metrics/`

## Output Format

Evaluation reports aggregate details from every inference result in the run:

```json
{
    "evaluation_metadata": {
        "run_id": "eval_YYYYMMMM_HHMMss",
        "evaluation_config": "path/to/originating_eval_config.json",
        "timestamp": "2026-01-28T20:21:27.870942",
        "pipeline_type": "full",
        "total_models": 1,
        "total_assessments": 1,
        "total_execution_time": 97.16061501
    },
    "model_results": [
        {
            "model_name": "LLaMa 3.2 1B",
            "assessments": [
                {
                    "name": "mmlu_simple_test",
                    "config": "path/to/assessment_configs/assessment_config.json",
                    "execution_time": 93.302893538028,
                    "total_prompts": 5,
                    "metric_summaries": {
                        "MMLU_Accuracy": {
                            "counts": {
                                "total_items": 5,
                                "scored_items": 5,
                                "skipped_items": 0,
                                "failed_items": 0,
                                "correct_answers": 1,
                                "incorrect_answers": 4
                            },
                            "scores": {
                                "accuracy": 0.2,
                                "accuracy_percentage": 20.0
                            },
                            "issues": []
                        }
                    },
                    "prompt_results": [
                        {
                            "name": "mmlu_simple_test_0",
                            "model_output": "model output text",
                            "inference_time": 3.8577214690158144,
                            "source_file": "/path/to/inference_result_file.json",
                            "gt_text": "C",
                            "metric_details": {
                                "MMLU_Accuracy": {
                                    "status": "ok",
                                    "errors": [],
                                    "correct": true
                                }
                            }
                        },
                        {...}
                    ]
                }
            ]
        }
    ]
}
```

---

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
    - Check model weight paths in model files
    - Ensure sufficient system memory
    - Verify CUDA/GPU setup if using GPU acceleration
2. **Permission Errors**
    - Ensure write permissions for `./logs/` and `./results/` directories
    - Check file paths in configuration files
3. **Configuration Validation Errors**
    - Verify JSON syntax in configuration files
    - Ensure all referenced prompt files exist
    - Check that model names match available models exactly

## Requirements

See `requirements.txt` for complete dependency list. Key requirements:

- Python 3.7+
- PyTorch
- Transformers
- Pydantic
- prompt_toolkit
- GPUtil (for GPU monitoring)
- psutil (for system monitoring)
