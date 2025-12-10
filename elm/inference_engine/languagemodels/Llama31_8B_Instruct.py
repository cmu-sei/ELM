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
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from gc import collect
from torch.cuda import empty_cache, is_available
from torch.backends import mps


class Model(LanguageModel):

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self._name = "LLaMa 3.1 8B Instruct"
        self.weights_dir = (
            "/path/to/your/weights/file"
        )
        self.tokenizer_dir = (
            "/path/to/your/tokenizer/file"
        )
        self.cache_dir = (
            "/path/to/your/cache"
        )

        if is_available():
            self.device = "cuda"
        elif mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            use_fast=True
        )
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.weights_dir,
            quantization_config=bnb_config
        )
        model.eval()
        self.model = model

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

    def ask(self, prompt, history=None, max_new_tokens=256):
        if history:
            preface = self.gen_history(history)
            prompt = preface + prompt

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            return_dict_in_generate=True,
        )
        token = outputs.sequences[0, input_length:]
        return self.tokenizer.decode(token, skip_special_tokens=True)

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
