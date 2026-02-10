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
import json
from elm.inference_engine.Inference_Engine import Inference_Engine
from elm.inference_engine.pydanticmodels import InferenceConfig, InferenceSet
from elm.common.pydantic_models import ModelSpec
from pydantic import ValidationError
from transformers import GenerationConfig


class TestPydanticModels:
    """Test Pydantic model validation"""
    
    def test_model_spec_valid_simple(self):
        """Test creating ModelSpec with just name"""
        spec = ModelSpec(name="Test Model")
        assert spec.name == "Test Model"
        assert spec.hyperparameters is None
        assert spec.quantization_config is None
    
    def test_model_spec_valid_with_hyperparams(self):
        """Test creating ModelSpec with hyperparameters"""
        spec = ModelSpec(
            name="Test Model",
            hyperparameters={"temperature": 0.7}
        )
        assert spec.name == "Test Model"
        assert spec.hyperparameters["temperature"] == 0.7
    
    def test_model_spec_missing_name(self):
        """Test ModelSpec requires name field"""
        with pytest.raises(ValidationError, match="name"):
            ModelSpec(hyperparameters={"temperature": 0.7})
    
    def test_inference_set_valid(self):
        """Test creating valid InferenceSet"""
        inference_set = InferenceSet(
            models=[ModelSpec(name="Model1")],
            prompts=["test.json"],
            hyperparameters={"temperature": 0.1}
        )
        assert len(inference_set.models) == 1
        assert len(inference_set.prompts) == 1
        assert len(inference_set.hyperparameters) == 1
    
    def test_inference_set_empty_models(self):
        """Test InferenceSet rejects empty models list"""
        with pytest.raises(ValidationError, match="Inference set must specify at least one model, it cannot be empty"):
            InferenceSet(models=[], prompts=["test.json"])
    
    def test_inference_set_empty_prompts(self):
        """Test InferenceSet rejects empty prompts list"""
        with pytest.raises(ValidationError, match="Inference set must specify at least one prompt file, it cannot be empty"):
            InferenceSet(
                models=[ModelSpec(name="Model1")],
                prompts=[]
            )
    
    def test_inference_config_valid(self):
        """Test creating valid InferenceConfig"""
        config = InferenceConfig(
            output_directory="./results",
            environment_config="dummy_env.json",
            inference_sets=[
                InferenceSet(
                    models=[ModelSpec(name="Model1")],
                    prompts=["test.json"]
                )
            ]
        )
        assert config.output_directory == "./results"
        assert len(config.inference_sets) == 1
    
    def test_inference_config_with_global_hyperparams(self):
        """Test InferenceConfig with global hyperparameters"""
        config = InferenceConfig(
            output_directory="./results",
            environment_config="dummy_env.json",
            hyperparameters={"temperature": 0.7},
            inference_sets=[
                InferenceSet(
                    models=[ModelSpec(name="Model1")],
                    prompts=["test.json"]
                )
            ]
        )
        assert config.hyperparameters["temperature"] == 0.7
    
    def test_inference_config_extra_fields_forbidden(self):
        """Test that extra fields are rejected"""
        with pytest.raises(ValidationError):
            InferenceConfig(
                output_directory="./results",
                environment_config="dummy_env.json",
                inference_sets=[],
                unknown_field="should fail"
            )


class TestHyperparameterMerging:
    """Test hyperparameter merging logic"""
    
    def test_merge_dicts_simple(self):
        """Test basic dictionary merging"""
        global_params = {"temperature": 0.7, "max_new_tokens": 256}
        set_params = {"temperature": 0.5}
        
        merged = {**global_params, **set_params}
        
        assert merged["temperature"] == 0.5  # Override
        assert merged["max_new_tokens"] == 256  # From global
    
    def test_merge_three_levels(self):
        """Test three-level merge: global -> set -> model"""
        global_params = {"temperature": 0.7, "max_new_tokens": 256, "top_p": 0.9}
        set_params = {"temperature": 0.5}
        model_params = {"max_new_tokens": 512}
        
        merged = {**global_params, **set_params, **model_params}
        
        assert merged["temperature"] == 0.5  # From set
        assert merged["max_new_tokens"] == 512  # From model
        assert merged["top_p"] == 0.9  # From global
    
    def test_merge_with_none(self):
        """Test merging when some dicts are None"""
        global_params = {"temperature": 0.7}
        set_params = None
        model_params = {"max_new_tokens": 256}
        
        merged = {**global_params, **(set_params or {}), **(model_params or {})}
        
        assert merged["temperature"] == 0.7
        assert merged["max_new_tokens"] == 256


class TestModelValidation:
    """Test model-level hyperparameter validation"""
    
    def test_generation_config_valid_params(self):
        """Test that valid hyperparameters are accepted"""
        config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9
        )
        
        assert config.max_new_tokens == 512
        assert config.temperature == 0.3
    
    def test_invalid_param_name_detection(self):
        """Test that our manual validation catches invalid parameter names"""
        valid_params = set(GenerationConfig().to_dict().keys())
        test_params = {"temprature": 0.7}  # Typo
        
        invalid_params = set(test_params.keys()) - valid_params
        assert invalid_params == {"temprature"}
    
    def test_generation_config_invalid_type(self):
        """Test that wrong types raise error"""
        with pytest.raises((ValueError, TypeError)):
            GenerationConfig(max_new_tokens="256")  # String instead of int

if __name__ == "__main__":
    pytest.main([__file__, "-v"])