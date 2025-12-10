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
from openai import OpenAI
import os


class Model(LanguageModel):

    _weights_are_local = False
    
    def __init__(self):
        self._name = "OpenAI o4 Mini"
        self.model = "o4-mini"

    def load(self):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        
    def gen_history(self, history):
        history_messages = ""
        for pair in history:
            history_messages += f'PROMPT: {pair[0]}\nRESPONSE: {pair[1]}\n'
        preface = f'Given this prompt history and context:\n<HISTORY>:\n{history_messages}\n<ENDHISTORY>\nRespond to the following query:\n'
        return preface

    def ask(self, prompt, history=None):
        if history:
            preface = self.gen_history(history)
            prompt = preface + prompt
        
        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )
        return response.output_text

    def delete(self):
        return True

    @property
    def name(self):
        return self._name

    def log(self):
        pass

    def prompter(self):
        pass
