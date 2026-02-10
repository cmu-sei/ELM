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
from elm.inference_engine.Inference_Engine import Inference_Log
import os
import json
import time


def test_update_log():
    log = Inference_Log()
    log.update_log({"model_name": "LLaMa 3.2 1B"})
    assert log.details["model_name"] == "LLaMa 3.2 1B"


def test_write_file(tmp_path):
    log = Inference_Log(output_dir=str(tmp_path))
    log.update_log({"model_name": "LLaMa 3.2 1B"})
    success, filepath = log.write_to_file()
    # Verify success
    assert success
    assert os.path.exists(filepath)
    # Verify filename format
    assert "LLaMa_3.2_1B" in filepath
    assert filepath.endswith(".json")
    # Verify content
    with open(filepath) as f:
        data = json.load(f)
        assert data["model_name"] == "LLaMa 3.2 1B"
        assert "start_time" in data
        assert "hardware_metrics" in data


def test_metrics_collection():
    engine = Inference_Engine()
    engine.start_metrics_collection("test")
    time.sleep(2.0)
    metrics = engine.stop_metrics_collection()
    assert "cpu" in metrics
    assert "gpu" in metrics
    assert "ram" in metrics

    # Verify CPU metrics have data
    assert "samples" in metrics["cpu"]
    assert len(metrics["cpu"]["samples"]) > 0
    assert metrics["cpu"]["samples"][0]["operation"] == "test"
    assert "avg_percent" in metrics["cpu"]
    assert "max_percent" in metrics["cpu"]
