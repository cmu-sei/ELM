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
from typing import List, Dict


class EnvironmentConfig(BaseModel):
    name: str
    models: List[Dict[str, str]]

    @validator("name")
    def name_must_meet_length_requirements(cls, name):
        min_length = 1
        if len(name) < min_length:
            raise ValueError(f"Name length must be at least {min_length}.")
        return name

    @validator("models")
    def models_must_be_valid(cls, models):
        for model in models:
            fields = ["model_name", "model_family"]
            for field in fields:
                if field in model and len(model[field]) > 0:
                    value = model[field]
                    if not isinstance(value, str):
                        raise ValueError(
                            f"Test set lists must only contain strings - invalid {field} detected: {value}"
                        )
                else:
                    raise ValueError(
                        f"Test sets must contain at least one list entry for every required field: {fields}"
                    )
        return models

