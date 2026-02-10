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

from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, Optional


class ModelSpec(BaseModel):
    """
    Specification for a model with optional hyperparameter and quantization overrides.
    
    Used by both InferenceEngine and EvaluationEngine to specify which model to run
    and with what configuration.
    
    Attributes:
        name: The model name (must match available models)
        hyperparameters: Optional dict of generation hyperparameters to override defaults
        quantization_config: Optional dict of quantization settings for model loading
    """
    model_config = ConfigDict(extra="forbid")

    name: str
    hyperparameters: Optional[Dict[str, Any]] = None
    quantization_config: Optional[Dict[str, Any]] = None