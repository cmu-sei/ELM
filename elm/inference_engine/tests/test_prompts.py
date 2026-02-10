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

from elm.inference_engine.Inference_Engine import Inference_Engine


def test_load_prompts_from_file():
    engine = Inference_Engine()
    prompt_file = "../prompts/prompt_mmlu_global_facts_test_few_shot.json"
    prompts = engine.load_prompts_from_file(prompt_filename=prompt_file)
    # Testing that for an individual file,
    # the entire file gets loaded correctly
    prompttext = "People tend to be optimistic about their own future"
    assert len(prompts) == 100
    assert prompts[0].name == "mmlu_global_facts_test_0"
    assert prompts[0].style == "basic"
    assert prompttext in prompts[0].text


def test_load_prompts_from_dir():
    # Testing that the prompts get loaded when an entire directory is passed
    engine = Inference_Engine()
    prompt_dir = "./prompts"
    prompts = engine.load_prompts_from_dir(promptdir=prompt_dir)
    prompttext = "People tend to be optimistic about their own future"
    assert len(prompts) == 100
    assert prompts[0].name == "mmlu_global_facts_test_0"
    assert prompts[0].style == "basic"
    assert prompttext in prompts[0].text


def test_populate_prompt_list():
    # Testing that the populate prompts list function works
    engine = Inference_Engine()
    prompt_dir = "./prompts"
    engine.populate_prompt_list(promptdir=prompt_dir)
    assert (
        engine.available_prompts["mmlu_global_facts_test_0"].name
        == "mmlu_global_facts_test_0"
    )
