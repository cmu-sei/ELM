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

from elm.inference_engine import languagemodels
from elm.inference_engine import pydanticmodels
import os
import sys
import json
import argparse
from datetime import datetime
from time import perf_counter
from importlib import import_module
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from pydantic import ValidationError
import psutil
import GPUtil
import threading
import time


# This class manages and formats logging of model output and performance metrics
class Inference_Log:

    def __init__(self, output_dir="./results"):
        self.details = {
            "start_time": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            "model_name": None,
            "prompt_config": None,
            "generation_config": None,
            "quantization_config": None,
            "load_time": None,
            "inference_time": None,
            "inference_results": None,
            "num_previous_prompts": 0,
            "hardware_metrics": {
                "load": None,
                "inference": {
                    "cpu": {"avg_percent": None, "max_percent": None, "samples": []},
                    "ram": {
                        "avg_used_gb": None,
                        "max_used_gb": None,
                        "avg_percent": None,
                        "max_percent": None,
                        "samples": [],
                    },
                    "gpu": [],  # Will contain metrics for each GPU
                },
            },
        }

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # Receives a K:V dictionary and if K is in self.details, update V
    # For updating nested dictionaries like `hardware_metrics`, should contain only
    # the sub-keys that need updating (e.g., {'hardware_metrics': {'load': metrics_data}})
    def update_log(self, key_value_dict):
        for key, value in key_value_dict.items():
            if key == "hardware_metrics" and isinstance(
                value, dict
            ):  # Check for hardware metrics first
                # Handle nested hardware metrics
                for metric_type, metric_data in value.items():
                    if metric_type in self.details["hardware_metrics"]:
                        self.details["hardware_metrics"][metric_type] = metric_data
            elif key in self.details:  # Handle all other keys
                self.details[key] = value

    # Writes metrics and results to a preformatted file with name like "LLaMa_3.1_8B_Instruct_20250228_205836.json"
    def write_to_file(self):
        filepath = f"{self.output_dir}/{self.details['model_name'].replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filepath, "w") as outfile:
                json.dump(self.details, outfile, indent=4)
        except Exception as e:
            return False, e
        return True, filepath


class Inference_Engine:

    # ------------------------------------------- #
    # System Control, Logging, and Error Handling #
    # ------------------------------------------- #

    def __init__(
        self, logdir="./logs", promptdir=None, envdir=None, metrics_interval=1.0
    ):
        self.init_time = datetime.now()
        self.logdir = logdir
        self.bypass_logging = False
        self.log_event(
            "SYSTEM", f"----- SYSTEM STARTING -----\nSystem started at {self.init_time}"
        )

        # Loads all available models+names into a dictionary
        # NPS edit this section to make model families
        self.available_model_families = []
        klass_list = languagemodels.__all__
        for klass in klass_list:
            self.available_model_families.append(klass)
        self.available_models = {}

        # Set prompts directory - use override if provided, otherwise default
        self.promptdir = (
            promptdir if promptdir else self._get_default_prompts_directory()
        )
        self._validate_prompts_directory()
        self.available_prompts = {}

        # Set environment config directory - use override if provided, otherwise default
        self.envdir = envdir if envdir else self._get_default_env_directory()
        self._validate_env_directory()

        # Models and prompts selected for evaluation
        self.model_selection = []
        self.prompt_selection = []
        self.loaded_model = None

        # Metrics tracking
        self.metrics_interval = metrics_interval  # Seconds between metrics captures
        self.metrics_thread = None
        self.should_collect_metrics = False
        self.current_operation = None
        self.metrics_data = {"cpu": [], "ram": [], "gpu": []}

        # Storage for model loading metrics (to be reused across prompts)
        self.current_model_loading_metrics = None

    # Enable logging directly to chosen logfile
    def log_event(self, type, description):

        if not self.bypass_logging:
            try:
                logfile_date = datetime.now().strftime("%Y-%m-%d")
                logfile_name = f"{self.logdir}/{logfile_date}.txt"
                os.makedirs(self.logdir, exist_ok=True)

                with open(logfile_name, "a") as logfile:
                    log_text = f"{datetime.now()} :: {type} :: {description}\n"
                    logfile.write(log_text)

            except Exception as e:
                print(f"Error writing to log: {e}")
                print("WARNING:  System unable to record system logs.")
                choice = input("Do you wish to continue without system logs? (y/n):  ")
                if "n" in choice.lower():
                    sys.exit(1)
                else:
                    self.bypass_logging = True
                    print("Continuing system execution without recording system logs.")

    # Log manually throw exceptions and gracefully exit
    def raise_exception(self, error_message):
        self.log_event("ERROR", f"Exiting - {error_message}")
        self.log_event("SYSTEM", "----- SYSTEM STOPPED -----")
        raise Exception(f"Inference engine could not start - {error_message}")

    # Helper for handling errors
    def _handle_error(self, error_message, exit_on_fail=False):
        if exit_on_fail:
            self.raise_exception(error_message)
        else:
            self.log_event("ERROR", error_message)

    # Empty model and prompt selections
    def reset_inference_config(self):
        self.model_selection = []
        self.available_prompts = {}
        self.prompt_selection = []

    def print_step(self, message):
        print(f"----------\n{message}")

    # ------------------------------------- #
    # Hardware Usage Monitoring and Logging #
    # ------------------------------------- #

    # Function to collect system metrics
    def _collect_metrics(self):
        while self.should_collect_metrics:
            timestamp = datetime.now().isoformat()

            # Collect CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)

            # Collect RAM metrics
            ram = psutil.virtual_memory()
            ram_used_gb = ram.used / (1024**3)
            ram_total_gb = ram.total / (1024**3)
            ram_percent = ram.percent

            # Store metrics
            self.metrics_data["cpu"].append(
                {
                    "timestamp": timestamp,
                    "operation": self.current_operation,
                    "percent": cpu_percent,
                }
            )

            self.metrics_data["ram"].append(
                {
                    "timestamp": timestamp,
                    "operation": self.current_operation,
                    "used_gb": ram_used_gb,
                    "total_gb": ram_total_gb,
                    "percent": ram_percent,
                }
            )

            # Collect GPU metrics if available
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_data = []
                    for gpu in gpus:
                        gpu_data.append(
                            {
                                "id": gpu.id,
                                "name": gpu.name,
                                "used_vram_gb": gpu.memoryUsed / 1024,
                                "total_vram_gb": gpu.memoryTotal / 1024,
                                "vram_percent": (gpu.memoryUsed / gpu.memoryTotal)
                                * 100,
                                "gpu_util": gpu.load * 100,  # Convert to percentage
                            }
                        )

                    self.metrics_data["gpu"].append(
                        {
                            "timestamp": timestamp,
                            "operation": self.current_operation,
                            "gpus": gpu_data,
                        }
                    )
            except Exception as e:
                self.log_event("WARNING", f"Error collecting GPU metrics: {e}")

            # Log current usage
            self.log_event(
                "METRICS",
                f"{self.current_operation} | CPU: {cpu_percent:.1f}% | "
                f"RAM: {ram_used_gb:.2f}/{ram_total_gb:.2f} GB ({ram_percent:.1f}%)",
            )

            time.sleep(self.metrics_interval)

    # Start metrics collection
    def start_metrics_collection(self, operation_name):
        self.current_operation = operation_name
        self.should_collect_metrics = True

        # Reset metrics data for the new operation
        for key in self.metrics_data:
            self.metrics_data[key] = []

        self.metrics_thread = threading.Thread(target=self._collect_metrics)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()

        self.log_event("METRICS", f"Started metrics collection for: {operation_name}")

    # Stop metrics collection and get the results
    def stop_metrics_collection(self):
        if self.should_collect_metrics:
            self.should_collect_metrics = False
            if self.metrics_thread:
                self.metrics_thread.join(timeout=2.0)

            # Calculate summary statistics
            metrics_summary = {"cpu": {}, "ram": {}, "gpu": []}

            if self.metrics_data["cpu"]:
                cpu_values = [entry["percent"] for entry in self.metrics_data["cpu"]]
                metrics_summary["cpu"] = {
                    "avg_percent": sum(cpu_values) / len(cpu_values),
                    "max_percent": max(cpu_values),
                    "samples": self.metrics_data["cpu"],
                }

            if self.metrics_data["ram"]:
                ram_percent = [entry["percent"] for entry in self.metrics_data["ram"]]
                ram_used_gb = [entry["used_gb"] for entry in self.metrics_data["ram"]]
                metrics_summary["ram"] = {
                    "avg_percent": sum(ram_percent) / len(ram_percent),
                    "max_percent": max(ram_percent),
                    "avg_used_gb": sum(ram_used_gb) / len(ram_used_gb),
                    "max_used_gb": max(ram_used_gb),
                    "samples": self.metrics_data["ram"],
                }

            if self.metrics_data["gpu"]:
                # Process each GPU separately
                gpu_count = len(self.metrics_data["gpu"][0]["gpus"])

                for gpu_idx in range(gpu_count):
                    gpu_summary = {
                        "id": self.metrics_data["gpu"][0]["gpus"][gpu_idx]["id"],
                        "name": self.metrics_data["gpu"][0]["gpus"][gpu_idx]["name"],
                    }

                    vram_percent = [
                        entry["gpus"][gpu_idx]["vram_percent"]
                        for entry in self.metrics_data["gpu"]
                    ]
                    vram_used = [
                        entry["gpus"][gpu_idx]["used_vram_gb"]
                        for entry in self.metrics_data["gpu"]
                    ]
                    gpu_util = [
                        entry["gpus"][gpu_idx]["gpu_util"]
                        for entry in self.metrics_data["gpu"]
                    ]

                    gpu_summary.update(
                        {
                            "avg_vram_percent": sum(vram_percent) / len(vram_percent),
                            "max_vram_percent": max(vram_percent),
                            "avg_used_vram_gb": sum(vram_used) / len(vram_used),
                            "max_used_vram_gb": max(vram_used),
                            "avg_gpu_util": sum(gpu_util) / len(gpu_util),
                            "max_gpu_util": max(gpu_util),
                            "samples": [
                                {
                                    "timestamp": entry["timestamp"],
                                    "used_vram_gb": entry["gpus"][gpu_idx][
                                        "used_vram_gb"
                                    ],
                                    "vram_percent": entry["gpus"][gpu_idx][
                                        "vram_percent"
                                    ],
                                    "gpu_util": entry["gpus"][gpu_idx]["gpu_util"],
                                }
                                for entry in self.metrics_data["gpu"]
                            ],
                        }
                    )

                    metrics_summary["gpu"].append(gpu_summary)

            self.log_event(
                "METRICS", f"Stopped metrics collection for: {self.current_operation}"
            )
            self.current_operation = None

            return metrics_summary

        return None

    # -------------- #
    # Prompt Loading #
    # -------------- #

    # Load PromptConfigs from properly formatted .json files
    def load_prompts_from_file(self, prompt_filename, exit_on_fail=False):
        file_prompts = []

        if not prompt_filename.endswith(".json"):
            error_message = f"Prompt file '{prompt_filename}' is not a .json file. Only .json files are supported."
            self._handle_error(error_message, exit_on_fail)
            return file_prompts

        try:
            # Resolve path to configured prompts directory
            resolved_path = self._resolve_prompt_file(prompt_filename)
            self.log_event("SYSTEM", f"Loading prompts from: {resolved_path}")
            with open(resolved_path, "r") as file:
                file_data = json.load(file)
                # Files must contain properly formatted PromptConfigs -> see pydanticmodels/PromptConfig.py
                if isinstance(file_data, list):
                    for config in file_data:
                        try:
                            prompt_config = pydanticmodels.PromptConfig(**config)
                            file_prompts.append(prompt_config)
                        except ValidationError as e:
                            error_message = f"Validation error for {resolved_path}: {e}"
                            self._handle_error(error_message, exit_on_fail)
                else:
                    error_message = f"Prompt file {resolved_path} must contain a list of prompt configurations"
                    self._handle_error(error_message, exit_on_fail)

        except FileNotFoundError as e:
            error_message = f"Could not locate prompt file: {e}"
            self._handle_error(error_message, exit_on_fail)
        except json.JSONDecodeError as e:
            error_message = f"JSON decode error in {prompt_filename}: {e}"
            self._handle_error(error_message, exit_on_fail)

        return file_prompts

    # Iterate through a directory to look for and ingest files containing PromptConfigs
    def load_prompts_from_dir(self, promptdir=None):
        target_dir = promptdir if promptdir else self.promptdir
        self.log_event("SYSTEM", f"Ingesting prompts from {target_dir}")
        prompt_load_start = perf_counter()

        loaded_prompts = []
        try:
            for root, dirs, files in os.walk(target_dir):
                for filename in files:
                    if filename.endswith(".json"):
                        # Get relative path from the base prompts directory
                        file_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(file_path, target_dir)
                        new_prompts = self.load_prompts_from_file(relative_path)
                        loaded_prompts += new_prompts
        except OSError as e:
            error_message = f"Error reading prompts directory {target_dir}: {e}"
            self.raise_exception(error_message)

        self.log_event(
            "SYSTEM",
            f"Loaded {len(loaded_prompts)} prompts after {perf_counter() - prompt_load_start} seconds.",
        )
        return loaded_prompts

    # Adds prompt names to list of available prompts to choose from
    def populate_prompt_list(self, promptdir=None):
        prompt_configs = self.load_prompts_from_dir(promptdir)
        for prompt_config in prompt_configs:
            self.available_prompts[prompt_config.name] = prompt_config

    # Verifies that at least one prompt is available for inference
    def check_for_loaded_prompts(self):
        # Do not allow usage of engine if there are 0 prompts and/or 0 models
        if len(self.available_prompts) < 1:
            error_message = "Inference engine could not start - no prompts were successfully loaded."
            self.raise_exception(error_message)

    # ------------- #
    # Model Loading #
    # ------------- #

    def load_env_from_file(self, env_filename, exit_on_fail=False):

        env_config = None

        if not env_filename.endswith(".json"):
            error_message = f"Env file '{env_filename}' is not a .json file. Only .json files are supported."
            self._handle_error(error_message, exit_on_fail)

        try:
            # Resolve path to configured prompts directory
            resolved_path = self._resolve_env_file(env_filename)
            self.log_event("SYSTEM", f"Loading environment from: {resolved_path}")
            with open(resolved_path, "r") as file:
                file_data = json.load(file)
                # Files must contain properly formatted EnvironmentConfigs ->
                # see pydanticmodels/EnvironmentConfig.py
                try:
                    env_config = pydanticmodels.EnvironmentConfig(**file_data)
                except ValidationError as e:
                    error_message = f"Validation error for {resolved_path}: {e}"
                    self._handle_error(error_message, exit_on_fail)

        except FileNotFoundError as e:
            error_message = f"Could not locate prompt file: {e}"
            self._handle_error(error_message, exit_on_fail)
        except json.JSONDecodeError as e:
            error_message = f"JSON decode error in {env_filename}: {e}"
            self._handle_error(error_message, exit_on_fail)

        return env_config

    # Loads the model into memory based on model_name
    def load_selected_model(self, model_name, quantization_config=None):
        if model_name in self.available_models:
            start_time = perf_counter()
            self.log_event("EVENT", f"Loading model {model_name}")

            model = self.available_models[model_name]
            if model._weights_are_local:
                self.start_metrics_collection(f"Model Load - {model_name}")

            # Pass quant config overrides through. Model code decides whether to use or ignore
            model.load(quantization_config=quantization_config)
            self.loaded_model = model
            load_time = perf_counter() - start_time
            self.log_event(
                "EVENT",
                f"{model_name} loaded after {load_time} seconds.",
            )
            loading_metrics = self.stop_metrics_collection()
            self.current_model_loading_metrics = loading_metrics
            return load_time
        else:
            self.log_event("ERROR", f"{model_name} failed to load.")
            return None

    # --------------- #
    # CLI Interaction #
    # --------------- #

    # Interactive command line prompt to choose a model with tab selection
    def prompt_for_model_selection(self):
        self.log_event("EVENT", "Prompted for model selection")

        model_choices = [name for name in self.available_models]
        completer = WordCompleter(model_choices, ignore_case=True)

        selected_model = None
        while selected_model not in model_choices:
            selected_model = prompt("Select a model: ", completer=completer)
            print(f"-> You selected: {selected_model}")
            if selected_model not in model_choices:
                print(
                    f"-> Your selection ({selected_model}) was invalid.  Please select a model from the list."
                )

        self.model_selection.append(selected_model)
        self.log_event("EVENT", f"{selected_model} selected")

    # Interactive command line prompt to choose a prompt with tab selection
    def prompt_for_prompt_selection(self):
        self.log_event("EVENT", "Prompted for prompt selection")

        prompt_choices = [prompt for prompt in self.available_prompts]
        completer = WordCompleter(prompt_choices, ignore_case=True)

        selected_prompt = None
        while selected_prompt not in prompt_choices:
            selected_prompt = prompt("Select a prompt: ", completer=completer)
            print(f"-> You selected: {selected_prompt}")
            if selected_prompt not in prompt_choices:
                print(
                    f"-> Your selection ({selected_prompt}) was invalid.  Please select a prompt from the list."
                )

        self.prompt_selection.append(selected_prompt)
        self.log_event("EVENT", f"{selected_prompt} selected")

    # ---------------- #
    # System Execution #
    # ---------------- #

    # Use one loaded model to execute inference using the text from one prompt
    def execute_inference(self, model, prompt, hyperparameters=None):
        start_time = perf_counter()
        self.log_event(
            "EVENT", f"Executing inference with {model._name} and {prompt.name}."
        )
        if model._weights_are_local:
            self.start_metrics_collection(
                f"Model Inference - {model._name} + {prompt.name}"
            )
        answer, generation_config = model.ask(prompt=prompt.text, hyperparameters=hyperparameters)
        inference_time = perf_counter() - start_time
        self.log_event(
            "EVENT",
            f"Inference with {model._name} finished after {inference_time} seconds.",
        )
        inference_metrics = self.stop_metrics_collection()
        return answer, generation_config, inference_time, inference_metrics

    # Performs inference using all selected models and all selected prompts
    def execute_inference_pipeline(self, output_dir="./results"):
        results_files = []

        for model_spec in self.model_selection:
            model_name = model_spec["name"]
            hyperparameters = model_spec.get("hyperparameters", {})
            quant_config = model_spec.get("quantization_config")

            # Check to make sure model successful loads
            self.print_step(f"Loading {model_name}...")
            load_time = self.load_selected_model(model_name, quant_config)
            executed_prompts = 0
            if self.loaded_model and load_time:
                used_quant_config = self.loaded_model.quantization_config_used

                for prompt in self.prompt_selection:

                    # Create a new Inference Log for this model/prompt combination and update model name
                    inference_log = Inference_Log(output_dir)
                    inference_log.update_log({"model_name": model_name})

                    # Log prompt config
                    prompt_config = self.available_prompts[prompt]
                    inference_log.update_log({"prompt_config": prompt_config.export()})

                    # Log load time and metrics
                    inference_log.update_log(
                        {
                            "load_time": load_time,
                            "num_previous_prompts": executed_prompts,
                            "hardware_metrics": {
                                "load": self.current_model_loading_metrics
                            },
                        }
                    )

                    # Execute inference
                    self.print_step(
                        f"Executing inference: '{model_name}' with '{prompt}'..."
                    )
                    result, generation_config, inference_time, inference_metrics = self.execute_inference(
                        self.loaded_model, prompt_config, hyperparameters
                    )

                    # Log inference results
                    inference_log.update_log(
                        {
                            "generation_config": generation_config,
                            "quantization_config": used_quant_config,
                            "inference_results": result,
                            "inference_time": inference_time,
                            "hardware_metrics": {"inference": inference_metrics},
                        }
                    )

                    # Export inference log
                    success, inference_results = inference_log.write_to_file()
                    if not success:
                        error_message = (
                            f"Error encountered saving logfile - {inference_results}"
                        )
                        self.raise_exception(error_message)
                    else:
                        results_files.append(inference_results)
                        message = f"Results file saved: {inference_results}"
                        self.log_event("SYSTEM", message)
                        print(message)

                    # Increment the prompt counter for this model
                    executed_prompts += 1

                # Manually remove model from memory after all prompts have been run
                if self.loaded_model._weights_are_local:
                    self.start_metrics_collection(f"Model Unload - {model_name}")
                self.loaded_model.delete()
                self.loaded_model = None
                self.stop_metrics_collection()

                # Clear the stored loading metrics
                self.current_model_loading_metrics = None
        return results_files

    # Runs if config file specified in command line
    def execute_with_config(self, config_file):
        self.log_event("SYSTEM", f"Loading models and prompts from {config_file}")
        config_load_start = perf_counter()

        # Only works with .json files
        if config_file.endswith(".json"):
            try:
                with open(config_file, "r") as file:
                    file_data = json.load(file)

                # Files must contain properly formatted InferenceConfigs -> see pydanticmodels/InferenceConfig.py
                if isinstance(file_data, list):
                    for config in file_data:
                        inference_config = pydanticmodels.InferenceConfig(**config)
                        global_hyperparams = inference_config.hyperparameters or {} # add globals if exist, otherwise empty dict
                        output_dir = inference_config.output_directory
                        inference_sets = inference_config.inference_sets
                        environment_config_path = inference_config.environment_config

                        # Load environment config file, including available models
                        loaded_env_config = self.load_env_from_file(
                            environment_config_path
                        )

                        self.print_step("Processing config inference sets...")

                        # The name in the environment config file for the model family must
                        # exactly match the class name
                        for model in loaded_env_config.models:

                            if (
                                "model_family" not in model.keys()
                                or "model_name" not in model.keys()
                            ):
                                error_message = f"model_name and model_family are required for each model"
                                self.raise_exception(error_message)

                            klass = model["model_family"]
                            module = import_module(
                                "elm.inference_engine.languagemodels." + klass
                            )
                            available_model = module.Model(model)
                            self.available_models[model["model_name"]] = available_model

                        for inference_set in inference_sets:
                            # Use dictionary unpacking so set-scoped params override global-scope params
                            inference_set_hyperparams = {
                                **global_hyperparams,
                                **(inference_set.hyperparameters or {})
                            }


                            for model_spec in inference_set.models:
                                model_name = model_spec.name
                                # Use dictionary unpacking so model-scoped params override set-scoped params
                                model_hyperparams = {
                                    **inference_set_hyperparams,
                                    **(model_spec.hyperparameters or {})
                                }
                                quantization_config = model_spec.quantization_config

                                # Validate model exists
                                if model_name not in self.available_models:
                                    error_message = f"Model '{model_name}' specified in '{config_file}' does not match any available models."
                                    self.raise_exception(error_message)

                                # Store model spec
                                self.model_selection.append({
                                    "name": model_name,
                                    "hyperparameters": model_hyperparams,
                                    "quantization_config": quantization_config
                                })

                            self.print_step(f"Models selected: {self.model_selection}") 

                            # Load prompts
                            for prompt_file in inference_set.prompts:
                                loaded_prompt_configs = self.load_prompts_from_file(
                                    prompt_file, exit_on_fail=True
                                )
                                for prompt_config in loaded_prompt_configs:
                                    prompt_name = prompt_config.name
                                    self.available_prompts[prompt_name] = prompt_config
                                    self.prompt_selection.append(prompt_name)

                            self.check_for_loaded_prompts()
                            self.execute_inference_pipeline(output_dir)
                            self.reset_inference_config()
                else:
                    error_message = (
                        f"Invalid config file format: {config_file}\n"
                        f"Expected: A JSON array (list) containing configuration object(s)\n"
                        f"Got: {type(file_data).__name__}\n\n"
                        f"Note: The config must be wrapped in square brackets [ ] even for a single configuration."
                    )
                    self.raise_exception(error_message)

            except ValidationError as e:
                error_message = f"File failed pydantic validation:\n{e}"
                self.raise_exception(error_message)
            except FileNotFoundError as e:
                error_message = f"File could not be loaded:\n{e}"
                self.raise_exception(error_message)
            except json.JSONDecodeError:
                error_message = (
                    f"Config ({config_file}) could not be loaded due to JSONDecodeError"
                )
                self.raise_exception(error_message)

            self.log_event(
                "SYSTEM",
                f"Loading complete after {perf_counter() - config_load_start} seconds.",
            )
        else:
            error_message = f"Config file is not .json filetype: {config_file}"
            self.raise_exception(error_message)

    # Runs if no config file was specified
    def execute_with_cli(self, promptdir=None):
        self.populate_prompt_list(promptdir)
        self.check_for_loaded_prompts()
        self.prompt_for_model_selection()
        self.prompt_for_prompt_selection()
        self.execute_inference_pipeline()

    # Initial run function
    def run(self, config=None, promptdir=None):
        if config:
            self.execute_with_config(config)
        else:
            self.execute_with_cli(promptdir)

        self.print_step("Finished inference execution! Shutting down.")

    # --------------------------------- #
    # Path Resolution Helper Methods    #
    # --------------------------------- #

    def _get_default_prompts_directory(self):
        """Get the default inference engine prompts directory."""
        inference_engine_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(inference_engine_dir, "prompts")

    def _get_default_env_directory(self):
        """Get the default inference environment configs directory."""
        inference_engine_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(inference_engine_dir, "environment_configs")

    def _validate_prompts_directory(self):
        """Ensure the prompts directory exists and is accessible."""
        if not os.path.exists(self.promptdir):
            error_message = (
                f"Prompts directory not found: {self.promptdir}\n"
                f"Default location: {self._get_default_prompts_directory()}"
            )
            self.raise_exception(error_message)

        if not os.path.isdir(self.promptdir):
            error_message = f"Prompts path is not a directory: {self.promptdir}"
            self.raise_exception(error_message)

    def _validate_env_directory(self):
        """Ensure the environment configs directory exists and is accessible."""
        if not os.path.exists(self.envdir):
            error_message = (
                f"Environment config directory not found: {self.envdir}\n"
                f"Default location: {self._get_default_env_directory()}"
            )
            self.raise_exception(error_message)

        if not os.path.isdir(self.envdir):
            error_message = f"Environment config path is not a directory: {self.envdir}"
            self.raise_exception(error_message)

    def _resolve_prompt_file(self, prompt_filename):
        """
        Resolve a prompt filename to its absolute path in the configured prompts directory.
        Supports both flat files and subfolder organization.

        Args:
            prompt_filename: Filename or relative path (e.g., "test.json" or "benchmarks/mmlu/sociology.json")

        Returns:
            Absolute path to the prompt file
        """
        # Clean the filename to handle any path separators
        clean_filename = prompt_filename.lstrip(os.sep).lstrip("/")
        prompt_path = os.path.join(self.promptdir, clean_filename)

        if not os.path.exists(prompt_path):
            error_message = (
                f"Prompt file not found: {prompt_path}\n"
                f"Looking for: {clean_filename}\n"
                f"In directory: {self.promptdir}"
            )
            raise FileNotFoundError(error_message)

        return prompt_path

    def _resolve_env_file(self, env_filename):
        """
        Resolve an env config filename to its absolute path in the configured env configs directory.
        Supports both flat files and subfolder organization.

        Args:
            env_filename: Filename or relative path (e.g., "test.json" or "benchmarks/mmlu/sociology.json")

        Returns:
            Absolute path to the env config file
        """
        # Clean the filename to handle any path separators
        clean_filename = env_filename.lstrip(os.sep).lstrip("/")
        env_path = os.path.join(self.envdir, clean_filename)

        if not os.path.exists(env_path):
            error_message = (
                f"Environment config file not found: {env_path}\n"
                f"Looking for: {clean_filename}\n"
                f"In directory: {self.envdir}"
            )
            raise FileNotFoundError(error_message)

        return env_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Execute inference using one or more models paired with one or more prompts."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Name of .json file to use as configuration for this execution.",
    )

    args = parser.parse_args()
    engine = Inference_Engine(logdir="./logs")
    engine.run(config=args.config)

    engine.log_event(
        "SYSTEM", f"System finished at {datetime.now()}\----- SYSTEM STOPPED -----"
    )
    engine.log_event("SYSTEM", "----- SYSTEM STOPPED -----")
