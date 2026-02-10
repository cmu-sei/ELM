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

from .LanguageModel import LanguageModel
from transformers import (
    GenerationConfig,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from gc import collect
from torch.cuda import empty_cache, is_available
from torch.backends import mps


class Model(LanguageModel):

    def __init__(self, specs):
        self.tokenizer = None
        self.model = None
        self.quantization_config_used = None

        if 'weights_dir' not in specs.keys():
            error_message = (
                f"weights_dir is a required input for this model family"
                        )
            self.raise_exception(error_message)
        if 'tokenizer_dir' not in specs.keys():
            error_message = (
                f"tokenizer_dir is a required input for this model family"
                        )

        self._name = specs['model_name']
        self.weights_dir = specs['weights_dir']
        self.tokenizer_dir = specs['tokenizer_dir']

        if is_available():
            self.device = "cuda"
        elif mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def load(self, quantization_config=None):
        self.model = T5ForConditionalGeneration.from_pretrained(self.weights_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_dir)

    def gen_history(self, history):
        history_messages = ""
        for pair in history:
            history_messages += f"PROMPT: {pair[0]}\nRESPONSE: {pair[1]}\n"
        preface = (
            "Given this prompt history and context:\n"
            "<HISTORY>:\n"
            f"{history_messages}\n"
            "<ENDHISTORY>\n"
            f"Respond to the following query:\n"
        )
        return preface

    def ask(self, prompt, history=None, hyperparameters=None):
        generation_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.7,
            top_k=50
        )
        # Merge defaults with user provided hyperparameters
        if hyperparameters:
            # Validate user provided valid parameter names
            valid_params = set(GenerationConfig().to_dict().keys())
            invalid_params = set(hyperparameters.keys()) - valid_params
            if invalid_params:
                invalid_list = ", ".join(f"'{p}'" for p in sorted(invalid_params))
                raise TypeError(
                    f"Invalid hyperparameter(s) for model '{self._name}': {invalid_list}"
                )
            # Update generation config with user provided hyperparams
            try:
                generation_config.update(**hyperparameters)
            except (ValueError) as e:
                raise ValueError(
                    f"Invalid hyperparameters for model `{self._name}`: {e}"
                ) from e

        if history:
            preface = self.gen_history(history)
            prompt = preface + prompt

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
        )
        token = outputs.sequences[0, input_length:]
        return self.tokenizer.decode(token, skip_special_tokens=True), generation_config.to_dict()

    @property
    def name(self):
        return self._name

    def delete(self):
        del self.tokenizer
        del self.model
        collect()
        empty_cache()

    def log(self):
        pass

    def prompter(self):
        pass
