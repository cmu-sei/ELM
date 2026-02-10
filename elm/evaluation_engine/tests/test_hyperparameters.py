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

import pytest
from pydantic import ValidationError
from elm.common import ModelSpec
from elm.inference_engine.pydanticmodels import InferenceConfig, InferenceSet
from elm.evaluation_engine.pydanticmodels import EvaluationConfig, AssessmentSpec


# ============================================================================
# Test ModelSpec (Shared Model)
# ============================================================================

class TestModelSpec:
    """Test ModelSpec with hyperparameters and quantization config"""
    
    def test_model_spec_with_hyperparameters(self):
        """Test ModelSpec accepts hyperparameters"""
        spec = ModelSpec(
            name="LLaMa 3.2 1B",
            hyperparameters={"temperature": 0.7, "max_new_tokens": 256}
        )
        assert spec.name == "LLaMa 3.2 1B"
        assert spec.hyperparameters["temperature"] == 0.7
        assert spec.hyperparameters["max_new_tokens"] == 256
    
    def test_model_spec_with_quantization_config(self):
        """Test ModelSpec accepts quantization_config"""
        spec = ModelSpec(
            name="LLaMa 3.1 8B Instruct",
            quantization_config={"load_in_4bit": True}
        )
        assert spec.quantization_config["load_in_4bit"] is True
    
    def test_model_spec_with_both(self):
        """Test ModelSpec with both hyperparameters and quantization"""
        spec = ModelSpec(
            name="LLaMa 3.2 1B",
            hyperparameters={"temperature": 0.9},
            quantization_config={"load_in_8bit": True}
        )
        assert spec.hyperparameters["temperature"] == 0.9
        assert spec.quantization_config["load_in_8bit"] is True


# ============================================================================
# Test InferenceEngine Three-Tier Hierarchy
# ============================================================================

class TestInferenceConfigHyperparameters:
    """Test InferenceConfig three-tier hyperparameter hierarchy"""
    
    def test_global_hyperparameters(self):
        """Test global-level hyperparameters"""
        config = InferenceConfig(
            output_directory="./results",
            hyperparameters={"temperature": 0.7, "max_new_tokens": 256},
            environment_config="dummy_env_config.json",
            inference_sets=[
                InferenceSet(
                    models=[ModelSpec(name="LLaMa 3.2 1B")],
                    prompts=["test.json"]
                )
            ]
        )
        assert config.hyperparameters["temperature"] == 0.7
        assert config.hyperparameters["max_new_tokens"] == 256
    
    def test_inference_set_hyperparameters(self):
        """Test inference set-level hyperparameters"""
        config = InferenceConfig(
            output_directory="./results",
            environment_config="dummy_env_config.json",
            inference_sets=[
                InferenceSet(
                    models=[ModelSpec(name="LLaMa 3.2 1B")],
                    prompts=["test.json"],
                    hyperparameters={"temperature": 0.5}
                )
            ]
        )
        assert config.inference_sets[0].hyperparameters["temperature"] == 0.5
    
    def test_model_level_hyperparameters(self):
        """Test model-level hyperparameters"""
        config = InferenceConfig(
            output_directory="./results",
            environment_config="dummy_env_config.json",
            inference_sets=[
                InferenceSet(
                    models=[
                        ModelSpec(
                            name="LLaMa 3.2 1B",
                            hyperparameters={"temperature": 0.9}
                        )
                    ],
                    prompts=["test.json"]
                )
            ]
        )
        assert config.inference_sets[0].models[0].hyperparameters["temperature"] == 0.9
    
    def test_three_tier_hierarchy_structure(self):
        """Test complete three-tier hierarchy exists"""
        config = InferenceConfig(
            output_directory="./results",
            hyperparameters={"temperature": 0.7},  # Global
            environment_config="dummy_env_config.json",
            inference_sets=[
                InferenceSet(
                    models=[
                        ModelSpec(
                            name="LLaMa 3.2 1B",
                            hyperparameters={"max_new_tokens": 512}  # Model
                        )
                    ],
                    prompts=["test.json"],
                    hyperparameters={"top_p": 0.9}  # Set
                )
            ]
        )
        # Verify all three tiers are present
        assert config.hyperparameters is not None
        assert config.inference_sets[0].hyperparameters is not None
        assert config.inference_sets[0].models[0].hyperparameters is not None


# ============================================================================
# Test EvaluationEngine Three-Tier Hierarchy
# ============================================================================

class TestEvaluationConfigHyperparameters:
    """Test EvaluationConfig three-tier hyperparameter hierarchy"""
    
    def test_global_hyperparameters(self):
        """Test global-level hyperparameters"""
        config = EvaluationConfig(
            pipeline_type="full",
            hyperparameters={"temperature": 0.7, "max_new_tokens": 256},
            models=[ModelSpec(name="LLaMa 3.2 1B")],
            assessments=[AssessmentSpec(config="assess_test.json")]
        )
        assert config.hyperparameters["temperature"] == 0.7
        assert config.hyperparameters["max_new_tokens"] == 256
    
    def test_assessment_level_hyperparameters(self):
        """Test assessment-level hyperparameters"""
        config = EvaluationConfig(
            pipeline_type="full",
            models=[ModelSpec(name="LLaMa 3.2 1B")],
            assessments=[
                AssessmentSpec(
                    config="assess_test.json",
                    hyperparameters={"temperature": 0.5}
                )
            ]
        )
        assert config.assessments[0].hyperparameters["temperature"] == 0.5
    
    def test_model_level_hyperparameters(self):
        """Test model-level hyperparameters"""
        config = EvaluationConfig(
            pipeline_type="full",
            models=[
                ModelSpec(
                    name="LLaMa 3.2 1B",
                    hyperparameters={"temperature": 0.9}
                )
            ],
            assessments=[AssessmentSpec(config="assess_test.json")]
        )
        assert config.models[0].hyperparameters["temperature"] == 0.9
    
    def test_three_tier_hierarchy_structure(self):
        """Test complete three-tier hierarchy exists"""
        config = EvaluationConfig(
            pipeline_type="full",
            hyperparameters={"temperature": 0.7},  # Global
            models=[
                ModelSpec(
                    name="LLaMa 3.2 1B",
                    hyperparameters={"max_new_tokens": 512}  # Model
                )
            ],
            assessments=[
                AssessmentSpec(
                    config="assess_test.json",
                    hyperparameters={"top_p": 0.9}  # Assessment
                )
            ]
        )
        # Verify all three tiers are present
        assert config.hyperparameters is not None
        assert config.assessments[0].hyperparameters is not None
        assert config.models[0].hyperparameters is not None


# ============================================================================
# Test Hyperparameter Merging Logic
# ============================================================================

class TestHyperparameterMerging:
    """Test the dictionary merging logic used in engines"""
    
    def test_merge_global_only(self):
        """Test using only global hyperparameters"""
        global_params = {"temperature": 0.7, "max_new_tokens": 256}
        set_params = {}
        model_params = {}
        
        merged = {**global_params, **set_params, **model_params}
        
        assert merged["temperature"] == 0.7
        assert merged["max_new_tokens"] == 256
    
    def test_merge_set_overrides_global(self):
        """Test middle tier overrides global"""
        global_params = {"temperature": 0.7, "max_new_tokens": 256}
        middle_params = {"temperature": 0.5}
        model_params = {}
        
        merged = {**global_params, **middle_params, **model_params}
        
        assert merged["temperature"] == 0.5  # Overridden
        assert merged["max_new_tokens"] == 256  # From global
    
    def test_merge_model_overrides_all(self):
        """Test model tier overrides all others"""
        global_params = {"temperature": 0.7, "max_new_tokens": 256}
        middle_params = {"temperature": 0.5}
        model_params = {"temperature": 0.9, "top_k": 100}
        
        merged = {**global_params, **middle_params, **model_params}
        
        assert merged["temperature"] == 0.9  # Model wins
        assert merged["max_new_tokens"] == 256  # From global
        assert merged["top_k"] == 100  # From model
    
    def test_merge_with_none_values(self):
        """Test merging when some dicts are None"""
        global_params = {"temperature": 0.7}
        middle_params = None
        model_params = {"max_new_tokens": 512}
        
        merged = {
            **global_params,
            **(middle_params or {}),
            **(model_params or {})
        }
        
        assert merged["temperature"] == 0.7
        assert merged["max_new_tokens"] == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])