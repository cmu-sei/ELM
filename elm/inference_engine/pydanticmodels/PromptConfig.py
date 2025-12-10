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
from typing import Optional


class PromptConfig(BaseModel):
    name: str
    style: str
    text: str
    gt_text: Optional[str] = None
    gt_file: Optional[str] = None

    @validator("name")
    def name_must_meet_length_requirements(cls, name):
        min_length = 1
        if len(name) < min_length:
            raise ValueError(f"Name length must be at least {min_length}.")
        return name

    @validator("style")
    def style_must_be_valid(cls, style):
        styles = ["basic", "single_token", "multi_token"]
        if style.lower() not in styles:
            raise ValueError(
                f"Style must be one of the following: {', '.join(styles)}."
            )
        return style

    @validator("text")
    def text_must_meet_length_requirements(cls, text):
        min_length = 1
        if len(text) < min_length:
            raise ValueError(f"Text length must be at least {min_length}.")
        return text

    @validator("gt_text")
    def gt_text_must_be_valid_if_provided(cls, gt_text):
        if gt_text is not None and len(gt_text.strip()) == 0:
            raise ValueError("Ground truth text cannot be empty if provided.")
        return gt_text

    @validator("gt_file")
    def gt_file_must_be_valid_if_provided(cls, gt_file):
        if gt_file is not None and len(gt_file.strip()) == 0:
            raise ValueError("Ground truth file path cannot be empty if provided.")
        return gt_file

    def export(self):
        return self.__dict__
