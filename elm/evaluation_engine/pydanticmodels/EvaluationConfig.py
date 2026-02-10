# ELM Code Library
#
# Copyright 2025 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE
# MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO
# WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR
# MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL.
# CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT
# TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Licensed under a MIT-style license, please see license.txt or contact
# permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release
# and unlimited distribution.  Please see Copyright notice for non-US Government
# use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to
# its own license.
#
# DM25-1265

from pydantic import BaseModel, field_validator, ConfigDict
from typing import Optional, List, Union, Dict, Any
from elm.common import ModelSpec
from .InferenceResultsConfig import InferenceResultsConfig
import os
import json


class AssessmentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str # Filename of assessment config
    hyperparameters: Optional[Dict[str, Any]] = None


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline_type: str
    outdir: Optional[str] = None

    # Global hyperparameters
    hyperparameters: Optional[Dict[str, Any]] = None

    # Full pipeline fields
    environment_config: Optional[str] = None
    models: Optional[List[ModelSpec]] = None
    assessments: Optional[List[AssessmentSpec]] = None
    metrics: Optional[List[str]] = None
    inference_results: Optional[Union[List[str], InferenceResultsConfig]] = None

    @field_validator("pipeline_type")
    @classmethod
    def validate_pipeline_type(cls, pipeline_type):
        valid_types = ["full", "metrics_only"]
        if pipeline_type not in valid_types:
            raise ValueError(f"Invalid pipeline_type: {pipeline_type}. Must be one of: {valid_types}")
        return pipeline_type

    @field_validator("models")
    @classmethod
    def validate_models_field(cls, v, info):
        pipeline_type = info.data.get("pipeline_type")

        if pipeline_type == "metrics_only" and v is not None:
            print("Warning: 'models' field will be ignored for metrics_only pipeline")
        elif pipeline_type in ["full", "inference_only"]:
            if not v or len(v) == 0:
                raise ValueError(f"Missing required field 'models' for pipeline_type '{pipeline_type}'")

        return v

    @field_validator("assessments")
    @classmethod
    def validate_assessments_field(cls, v, info):
        pipeline_type = info.data.get("pipeline_type")

        if pipeline_type == "metrics_only" and v is not None:
            print("Warning: 'assessments' field will be ignored for metrics_only pipeline")
        elif pipeline_type in ["full", "inference_only"]:
            if not v or len(v) == 0:
                raise ValueError(f"Missing required field 'assessments' for pipeline_type '{pipeline_type}'")

        return v

    @field_validator("metrics")
    @classmethod
    def validate_metrics_field(cls, v, info):
        pipeline_type = info.data.get("pipeline_type")

        if pipeline_type == "metrics_only":
            if not v or len(v) == 0:
                raise ValueError("Missing required field 'metrics' for pipeline_type 'metrics_only'. Must specify a non-empty list of metrics to calculate.")

            for metric in v:
                if not isinstance(metric, str) or len(metric.strip()) == 0:
                    raise ValueError(f"All metrics must be non-empty strings, got: {metric}")

        return v

    @field_validator("inference_results")
    @classmethod
    def validate_inference_results_field(cls, v, info):
        pipeline_type = info.data.get("pipeline_type")

        if pipeline_type == "metrics_only":
            if v is None:
                raise ValueError("Missing required field 'inference_results' for pipeline_type 'metrics_only'")

            # If it's a list, validate that files exist
            if isinstance(v, list):
                if len(v) == 0:
                    raise ValueError("inference_results cannot be empty")

                # Validate each file path
                missing_files = []
                invalid_files = []

                for file_path in v:
                    if not isinstance(file_path, str):
                        raise ValueError(f"All inference result paths must be strings, got: {type(file_path)}")

                    # Expand user and environment variables
                    resolved_path = os.path.expandvars(os.path.expanduser(file_path))

                    if not os.path.exists(resolved_path):
                        missing_files.append(resolved_path)
                        continue

                    if not os.path.isfile(resolved_path):
                        invalid_files.append(f"{resolved_path} (not a file)")
                        continue

                    # Try to validate JSON structure
                    try:
                        with open(resolved_path, 'r') as f:
                            data = json.load(f)
                            if not isinstance(data, dict):
                                invalid_files.append(f"{resolved_path} (not a JSON object)")
                                continue
                            if "model_name" not in data:
                                print(f"Warning: {resolved_path} missing 'model_name' field")
                            if "inference_results" not in data:
                                print(f"Warning: {resolved_path} missing 'inference_results' field")
                    except json.JSONDecodeError as e:
                        invalid_files.append(f"{resolved_path} (invalid JSON: {e})")
                    except Exception as e:
                        invalid_files.append(f"{resolved_path} (read error: {e})")

                # Report all validation errors
                errors = []
                if missing_files:
                    errors.append(f"Missing files: {', '.join(missing_files)}")
                if invalid_files:
                    errors.append(f"Invalid files: {', '.join(invalid_files)}")

                if errors:
                    raise ValueError(f"Inference result file validation failed: {'; '.join(errors)}")

            # If it's an InferenceResultsConfig (dict), validation happens in that model
            elif isinstance(v, dict):
                # Convert dict to InferenceResultsConfig for validation
                try:
                    v = InferenceResultsConfig(**v)
                except Exception as e:
                    raise ValueError(f"Invalid inference_results configuration: {e}")

        return v

    def export(self):
        return self.__dict__