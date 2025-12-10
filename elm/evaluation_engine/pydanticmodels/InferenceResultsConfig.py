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

from pydantic import BaseModel, validator
from typing import Optional, List
import os
import json


class FilterConfig(BaseModel):
    """Configuration for filtering results from evaluation reports."""
    models: Optional[List[str]] = None
    assessments: Optional[List[str]] = None


class InferenceResultsConfig(BaseModel):
    """Configuration for structured inference results (dict format)."""
    type: str

    # For "from_evaluation_report" type
    report_path: Optional[str] = None
    filter: Optional[FilterConfig] = None

    @validator("type")
    def validate_type(cls, v):
        valid_types = ["from_evaluation_report"]
        if v not in valid_types:
            raise ValueError(f"Unsupported inference_results type: {v}. Must be one of: {valid_types}")
        return v

    @validator("report_path")
    def validate_report_path(cls, v, values):
        if values.get("type") == "from_evaluation_report":
            if not v:
                raise ValueError("report_path required for type 'from_evaluation_report'")

            # Expand path and check if it exists
            expanded_path = os.path.expandvars(os.path.expanduser(v))
            if not os.path.exists(expanded_path):
                raise ValueError(f"Evaluation report file not found: {expanded_path}")

            # Validate it's a valid JSON file
            try:
                with open(expanded_path, 'r') as f:
                    report = json.load(f)
                if "model_results" not in report:
                    raise ValueError("Evaluation report missing required field: 'model_results'")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in evaluation report: {e}")
            except Exception as e:
                raise ValueError(f"Error reading evaluation report: {e}")

        return v

    def export(self):
        return self.__dict__