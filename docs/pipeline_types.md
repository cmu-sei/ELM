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

## Pipeline Types

### Full Pipeline

Runs inference on specified models and prompts, then calculates metrics.

**When to use:**
- First-time evaluations
- Testing new prompts or models
- Comprehensive model comparison

**Configuration:**
```json
{
    "pipeline_type": "full",
    "environment_config": "test_env.json",
    "models": [
        {
            "name": "LLaMa 3.2 1B",
        },
        {
            "name": "LLaMa 3.2 3B"
        }
    ],
    "assessments": [
        {
            "config": "assess_mmlu_simple_test.json",
        }
    ],
    "outdir": "my_evaluation"
}
```

**Execution:**
```bash
python EvaluationEngine.py -c my_eval.json
```

### Metrics-Only Pipeline

Calculates metrics on existing inference results without re-running inference.

**When to use:**
- Trying different metrics on same inference results
- Adding newly-implemented metrics to historical data
- Reanalyzing subsets of previous runs

**Configuration (from evaluation report):**
```json
{
    "pipeline_type": "metrics_only",
    "metrics": ["ROUGE_Score"],
    "inference_results": {
        "type": "from_evaluation_report",
        "report_path": "/path/to/evaluation_report_{run_id}.json",
        "filter": {
            "models": ["LLaMa 3.2 1B"],
            "assessments": ["mmlu_simple_test"]
        }
    },
    "outdir": "reanalysis"
}
```

**Configuration (direct file list):**
```json
{
    "pipeline_type": "metrics_only",
    "metrics": ["MMLU_Accuracy"],
    "inference_results": [
        "/path/to/inference_result_1.json",
        "/path/to/inference_result_2.json"
    ],
    "outdir": "custom_analysis"
}
```