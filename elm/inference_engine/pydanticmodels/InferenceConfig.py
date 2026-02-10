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
from typing import List, Dict, Any, Optional
from elm.common import ModelSpec

class InferenceSet(BaseModel):
    """
    A set of models to run against a set of prompts.
    
    Attributes:
        models: List of model specifications
        prompts: List of prompt file paths
        hyperparameters: Optional dict to override global hyperparameters for this set
    """
    model_config = ConfigDict(extra="forbid")

    models: List[ModelSpec]
    prompts: List[str]
    hyperparameters: Optional[Dict[str, Any]] = None

    @field_validator("models")
    @classmethod
    def models_not_empty(cls, models: List[ModelSpec]) -> List[ModelSpec]:
        if len(models) == 0:
            raise ValueError("Inference set must specify at least one model, it cannot be empty")
        return models
    
    @field_validator("prompts")
    @classmethod
    def prompts_not_empty(cls, prompts: List[str]) -> List[str]:
        if len(prompts) == 0:
            raise ValueError("Inference set must specify at least one prompt file, it cannot be empty")
        return prompts
    
class InferenceConfig(BaseModel):
    """
    Top-level configuration for inference execution.
    
    Attributes:
        output_directory: Path where results will be saved
        inference_sets: List of model-prompt combinations to execute
        hyperparameters: Optional dict of global default hyperparameters
        environment_config: Path to environment config file
    """
    model_config = ConfigDict(extra="forbid")

    output_directory: str
    inference_sets: List[InferenceSet]
    hyperparameters: Optional[Dict[str, Any]] = None
    environment_config: str

    @field_validator("inference_sets")
    @classmethod
    def inference_sets_not_empty(cls, inference_sets: List[InferenceSet]) -> List[InferenceSet]:
        if len(inference_sets) == 0:
            raise ValueError("Must specify at least one inference set")
        return inference_sets
