from .LanguageModel import LanguageModel
from openai import OpenAI


class Model(LanguageModel):

    _weights_are_local = False

    def __init__(self, specs):
        if "model_code" not in specs:
            self.raise_exception("model_code is a required input for this model family")

        self._name = specs["model_name"]
        self.model = specs["model_code"]
        self.base_url = specs.get("base_url", "http://localhost:11434/v1")
        self.quantization_config_used = None

    def load(self, quantization_config=None):
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="ollama",
        )

    def gen_history(self, history):
        history_messages = ""
        for pair in history:
            history_messages += f"PROMPT: {pair[0]}\nRESPONSE: {pair[1]}\n"
        preface = f"Given this prompt history and context:\n<HISTORY>:\n{history_messages}\n<ENDHISTORY>\nRespond to the following query:\n"
        return preface

    def ask(self, prompt, history=None, hyperparameters=None):
        if history:
            preface = self.gen_history(history)
            prompt = preface + prompt

        generation_config = {}

        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text, generation_config

    def delete(self):
        return True

    @property
    def name(self):
        return self._name

    def log(self):
        pass

    def prompter(self):
        pass
