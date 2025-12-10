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

from pydantic import BaseModel, validator, ConfigDict
from typing import List, Dict


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_directory: str
    inference_sets: List[Dict[str, List[str]]]

    @validator("inference_sets")
    def inference_sets_must_be_valid(cls, inference_sets):
        for inference_set in inference_sets:
            fields = ["models", "prompts"]
            for field in fields:
                if field in inference_set and len(inference_set[field]) > 0:
                    for item in inference_set[field]:
                        if not isinstance(item, str):
                            raise ValueError(
                                f"Test set lists must only contain strings - invalid {field} detected: {item}"
                            )
                else:
                    raise ValueError(
                        f"Test sets must contain at least one list entry for every required field: {fields}"
                    )
        return inference_sets
