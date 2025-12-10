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

# Adding Custom Metrics

Guide for implementing custom metrics in the evaluation framework.

## Table of Contents

1. [Overview](#overview)
2. [MetricBase Interface](#metricbase-interface)
3. [Implementation Example](#implementation-example)
4. [Best Practices](#best-practices)

---

## Overview

Metrics in this framework calculate scores or perform analysis on model inference results. All metrics implement a common interface defined by `MetricBase`.

### When to Create a Custom Metric

- Implementing a new evaluation criterion (e.g., semantic similarity, factual accuracy)
- Calculating domain-specific scores
- Aggregating or analyzing results in custom ways
- Comparing outputs against ground truth with custom logic

### Metric Architecture

```
metrics/
├── MetricBase.py              # Abstract base class
├── MMLU_Accuracy.py           # Example: accuracy calculation
├── ROUGE_Score.py             # Example: text similarity
├── Your_Custom_Metric.py      # Your implementation
└── __init__.py                # Registration
```

---

## MetricBase Interface

All metrics must inherit from `MetricBase` and implement its abstract methods.

### Required Implementation

```python
from .MetricBase import MetricBase

class Your_Metric(MetricBase):

    @property
    def name(self):
        """Return unique metric identifier"""
        return "Your_Metric_Name"

    def compute(self, inference_results):
        """
        Calculate metric on inference results.

        Args:
            inference_results: List of inference result dictionaries

        Returns:
            dict with 'summary' and 'individual_results'
        """
        return {
            "summary": {
                "counts": {},
                "scores": {},
                "issues": issues,
            },
            "individual_results": individual_results
        }
```

### Input Format

The `compute()` method receives a list of inference result dictionaries:

```python
[
    {
        "model_name": "LLaMa 3.2 1B",
        "prompt_config": {
            "name": "prompt_1",
            "style": "basic",
            "text": "What is 2+2?",
            "gt_text": "4",
            "gt_file": null
        },
        "inference_results": "The answer is 4.",
        "inference_time": 5.2,
        "_source_file": "/path/to/result.json"
    },
    # ... more results
]
```

### Output Format

Return a dictionary with this structure:

```python
{
    "summary": {
        "counts": {
            "total_items": int,      # Total inference results processed
            "scored_items": int,     # Successfully scored
            "skipped_items": int,    # Skipped (e.g., missing ground truth)
            "failed_items": int      # Failed during calculation (optional)
        },
        "scores": {
            # Metric-specific aggregate scores
            "metric_score": float,
            "metric_percentage": float
        },
        "issues": [
            # List of problems encountered
            {
                "index": int,
                "type": "error_type",
                "description": "detailed description"
            }
        ]
    },
    "individual_results": [
        # Per-inference-result details
        {
            "prompt_name": "prompt_1",
            "status": "ok",  # or "skipped" or "failed"
            "errors": [],
            # Metric-specific fields
            "score": float,
            "correct": bool
        }
    ]
}
```

---

## Implementation Example

Let's implement a custom metric that calculates exact string match accuracy.

### Step 1: Create Metric File

Create `metrics/Exact_Match.py`:

```python
from .MetricBase import MetricBase


class Exact_Match(MetricBase):

    @property
    def name(self):
        return "Exact_Match"

    def compute(self, inference_results):
        """
        Compute exact string match accuracy between model output and ground truth.
        """
        total_items = len(inference_results) if inference_results else 0

        if total_items == 0:
            return self._empty_result()

        individual_results = []
        issues = []
        correct_count = 0
        skipped_count = 0

        for i, result in enumerate(inference_results):
            prompt_name = result.get("prompt_config", {}).get("name", "unknown")
            model_output = result.get("inference_results", "")
            ground_truth = result.get("prompt_config", {}).get("gt_text")

            # Validate inputs
            skip_reason = self._validate_inputs(model_output, ground_truth)

            if skip_reason:
                # Record skipped entry
                individual_results.append({
                    "prompt_name": prompt_name,
                    "status": "skipped",
                    "errors": [skip_reason]
                })
                issues.append({
                    "index": i,
                    "type": skip_reason,
                    "description": f"Skipped prompt {prompt_name}: {skip_reason}"
                })
                skipped_count += 1
            else:
                # Calculate exact match
                is_match = self._exact_match(model_output, ground_truth)
                if is_match:
                    correct_count += 1

                individual_results.append({
                    "prompt_name": prompt_name,
                    "status": "ok",
                    "errors": [],
                    "match": is_match,
                    "model_output": model_output.strip(),
                    "ground_truth": ground_truth.strip()
                })

        # Calculate summary statistics
        scored_items = total_items - skipped_count
        accuracy = correct_count / scored_items if scored_items > 0 else 0.0

        return {
            "summary": {
                "counts": {
                    "total_items": total_items,
                    "scored_items": scored_items,
                    "skipped_items": skipped_count,
                    "correct_matches": correct_count,
                    "incorrect_matches": scored_items - correct_count
                },
                "scores": {
                    "exact_match_accuracy": accuracy,
                    "exact_match_percentage": accuracy * 100
                },
                "issues": issues
            },
            "individual_results": individual_results
        }

    def _exact_match(self, model_output, ground_truth):
        """Check if strings match exactly (case-insensitive, whitespace-stripped)"""
        output_clean = str(model_output).strip().lower()
        truth_clean = str(ground_truth).strip().lower()
        return output_clean == truth_clean

    def _validate_inputs(self, model_output, ground_truth):
        """Validate that required inputs are present"""
        if not model_output or not str(model_output).strip():
            return "empty_model_output"
        if ground_truth is None or not str(ground_truth).strip():
            return "empty_ground_truth"
        return None

    def _empty_result(self):
        """Return empty result structure when no inference results provided"""
        return {
            "summary": {
                "counts": {
                    "total_items": 0,
                    "scored_items": 0,
                    "skipped_items": 0
                },
                "scores": {},
                "issues": []
            },
            "individual_results": []
        }


# Factory function for metric instantiation
def Metric():
    return Exact_Match()
```

### Step 2: Register Metric

Add to `metrics/__init__.py`:

```python
from . import Exact_Match

__all__ = [
    "MMLU_Accuracy",
    "ROUGE_Score",
    "Exact_Match"  # Add your metric here
]
```

### Step 3: Use in Evaluation

Add to assessment config:

```json
{
    "name": "my_assessment",
    "prompts": ["../inference_engine/prompts/my_prompts.json"],
    "metrics": ["Exact_Match"]
}
```

---

## Best Practices

### Input Validation

Always validate inputs before processing:

```python
def _validate_inputs(self, model_output, ground_truth):
    """Validate required fields"""
    errors = []

    if not model_output or not str(model_output).strip():
        errors.append("empty_model_output")

    if ground_truth is None:
        errors.append("missing_ground_truth")
    elif not str(ground_truth).strip():
        errors.append("empty_ground_truth")

    return errors[0] if errors else None
```

### Error Handling

Wrap calculations in try-except to handle unexpected errors:

```python
def compute(self, inference_results):
    # ... initialization ...

    for i, result in enumerate(inference_results):
        try:
            # Calculation logic
            score = self._calculate_score(result)
            individual_results.append({
                "status": "ok",
                "score": score
            })
        except Exception as e:
            # Record failure
            individual_results.append({
                "status": "failed",
                "errors": [f"calculation_error: {str(e)}"]
            })
            issues.append({
                "index": i,
                "type": "calculation_error",
                "description": str(e)
            })
```

### Consistent Output Structure

Use helper methods for consistency:

```python
def _create_success_result(self, prompt_name, **metric_fields):
    """Create successful result entry"""
    return {
        "prompt_name": prompt_name,
        "status": "ok",
        "errors": [],
        **metric_fields
    }

def _create_skipped_result(self, prompt_name, reason):
    """Create skipped result entry"""
    return {
        "prompt_name": prompt_name,
        "status": "skipped",
        "errors": [reason]
    }

def _create_failed_result(self, prompt_name, error):
    """Create failed result entry"""
    return {
        "prompt_name": prompt_name,
        "status": "failed",
        "errors": [f"calculation_error: {error}"]
    }
```

### Summary Statistics

Include meaningful aggregate scores:

```python
# Basic counts
"counts": {
    "total_items": len(inference_results),
    "scored_items": successful_count,
    "skipped_items": skipped_count,
    "failed_items": failed_count
}

# Example metric-specific scores
"scores": {
    "mean_score": sum(scores) / len(scores),
    "median_score": sorted(scores)[len(scores) // 2],
    "min_score": min(scores),
    "max_score": max(scores),
    "std_dev": calculate_std_dev(scores)
}
```