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

from pydantic import BaseModel, field_validator
from typing import List, Optional

class AssessmentConfig(BaseModel):
    name: str
    description: Optional[str] = None
    version: Optional[str] = None
    prompts: List[str]
    metrics: List[str]

    config_path: Optional[str] = None

    @field_validator("prompts")
    @classmethod
    def validate_prompts_not_empty(cls, prompts):
        if not prompts:
            raise ValueError("Assessment config must specify at least one prompt file")
        return prompts
    
    @field_validator("metrics")
    @classmethod
    def validate_metrics_not_empty(cls, metrics):
        if not metrics:
            raise ValueError("Assessment must specify at least one metric")
        return metrics