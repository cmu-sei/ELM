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

### 1. Install Dependencies

After cloning the repository, navigate to the root of the repository and run:
```bash
pip install -e .
```
This will install the code as a Python package and install all requirements.

See `requirements.txt` for a complete list of dependencies which will be installed.

### 2. Configure Model Paths

Update model weight and tokenizer paths in `elm/inference_engine/languagemodels/` to match your environment.

Add new model files as needed.

### 3. Run Evaluation

```bash
cd elm/evaluation_engine
python EvaluationEngine.py -c example_evaluation_config.json
```

This runs a full pipeline: inference on prompts → calculate metrics → generate report.

## Components

```
elm/
├── inference_engine/          # Model management and inference
│   ├── languagemodels/        # Model implementations (plugin system)
│   ├── prompts/               # Prompt configuration files
│   └── Inference_Engine.py   # Core inference orchestration
│
└── evaluation_engine/         # Evaluation orchestration
    ├── metrics/               # Metric implementations (plugin system)
    ├── pydanticmodels/        # Configuration validation schemas
    ├── assessment_configs/    # Assessment definitions
    ├── evaluation_configs/    # Evaluation specifications
    ├── evaluation_results/    # Generated outputs
    └── EvaluationEngine.py    # Core evaluation orchestration
```

### Configuration Hierarchy

```
Evaluation Config (Top Level)
├── Specifies: pipeline type, models, assessments, metrics
├── References: Assessment configs OR inference results
│
└─> Assessment Config (Mid Level)
    ├── Specifies: assessment name, prompt files, metrics
    ├── References: Prompt config files
    │
    └─> Prompt Config (Bottom Level)
        └── Contains: individual prompts with optional ground truth
```
