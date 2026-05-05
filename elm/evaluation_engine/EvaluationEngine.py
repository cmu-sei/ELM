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

import os
import sys
import json
import argparse
from datetime import datetime
from time import perf_counter
from importlib import import_module
from pydantic import ValidationError
from pathlib import Path
from elm.inference_engine.Inference_Engine import Inference_Engine
from elm.evaluation_engine import pydanticmodels
from elm.evaluation_engine.pydanticmodels.InferenceResultsConfig import InferenceResultsConfig
import elm.evaluation_engine.metrics as metrics


class Evaluation_Engine:
    def __init__(self, 
                 evaluation_configs_dir=None, 
                 assessment_configs_dir=None, 
                 results_dir=None,
                 promptdir=None):
        self.init_time = datetime.now()

        # Set up directory paths with defaults or overrides if provided
        self._setup_directories(evaluation_configs_dir, assessment_configs_dir, results_dir)

        # Available components
        self.available_metrics = {}
        self.available_models = {}

        # Current evaluation state
        self.evaluation_config = None
        self.promptdir = promptdir

        # Initialize available metrics
        self._get_available_metrics()

        print(f"Evaluation Engine initialized at {self.init_time}")
        print(f"Available metrics: {list(self.available_metrics.keys())}")

    # ------------------------------------------- #
    #     Public Interface / Callable Methods     #
    # ------------------------------------------- #

    def run(self, config_file=None):
        if not config_file:
            raise ValueError("Config file is required for evaluation engine")

        start_time = perf_counter()

        try:
            self.load_evaluation_config(config_file)
            self.config_file_path = config_file

            # Generate run_id for this evaluation run
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"eval_{run_timestamp}"

            # Resolve base output directory path
            configured_outdir = self.evaluation_config.outdir or self.results_dir
            configured_outdir = os.path.expandvars(os.path.expanduser(configured_outdir))
            base_output_dir = Path(configured_outdir)
            if not base_output_dir.is_absolute():
                base_output_dir = Path(self.results_dir) / base_output_dir
            base_output_dir = base_output_dir.resolve()
            
            # Create run-specific directory
            output_dir = base_output_dir / f"run_{self.run_id}"

            # Execute specified pipeline
            pipeline_type = self.evaluation_config.pipeline_type
            print(f"Executing {pipeline_type} pipeline...")

            if pipeline_type == "full":
                evaluation_report = self.execute_full_pipeline(output_dir)
            elif pipeline_type == "metrics_only":
                evaluation_report = self.execute_metrics_only_pipeline(output_dir)
            elif pipeline_type == "inference_only":
                evaluation_report = self.execute_inference_only_pipeline(output_dir)
            else:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")

            execution_time = perf_counter() - start_time
            evaluation_report["evaluation_metadata"]["total_execution_time"] = round(execution_time, 1)

            report_file = self.save_evaluation_report(evaluation_report, output_dir)

            print(f"Evaluation finished after {execution_time:.2f} seconds")  # TODO: log
            print(f"Evaluation complete! Report saved to: {report_file}")

        except Exception as e:
            print(f"Evaluation failed: {e}")
            raise

    def load_evaluation_config(self, config_file):
        """
        Load and validate evaluation configuration file using Pydantic.
        """
        if not config_file.endswith(".json"):
            raise ValueError(f"Config file must be .json: {config_file}")

        try:
            # Resolve config file path
            resolved_path = self._resolve_config_filepath(config_file, self.evaluation_configs_dir)
            with open(resolved_path, "r") as file:
                config_data = json.load(file)

            # Validate using Pydantic model
            try:
                self.evaluation_config = pydanticmodels.EvaluationConfig(**config_data)
            except ValidationError as e:
                raise ValueError(f"Invalid evaluation config: {e}")

            print(f"Loaded evaluation config: {resolved_path}")
            return self.evaluation_config

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file not found: {str(e)}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in config file: {config_file}")

    def load_assessment_config(self, filename):
        """
        Load and validate assessment configuration file using Pydantic.
        """
        try:
            # Resolve config file path
            resolved_path = self._resolve_config_filepath(filename, self.assessment_configs_dir)
            with open(resolved_path, "r") as file:
                config_data = json.load(file)

            try:
                config_data['config_path'] = resolved_path
                assessment_config = pydanticmodels.AssessmentConfig(**config_data)
                print(f"Loaded assessment config: {resolved_path}")
                return assessment_config
            except ValidationError as e:
                raise ValueError(f"Invalid assessment config in {filename}: {e}")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Assessment config not found: {str(e)}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in assessment config: {filename}")

    # ------------------------------------------- #
    #              Pipeline Execution             #
    # ------------------------------------------- #

    def execute_full_pipeline(self, output_dir):
        """
        Execute the full evaluation pipeline (inference + metrics).
        """
        print("Starting full evaluation pipeline...")

        evaluation_report = self._create_evaluation_report_structure(
            total_models=len(self.evaluation_config.models),
            total_assessments=len(self.evaluation_config.assessments),
        )

        # Process each model (ModelSpec object)
        for model_spec in self.evaluation_config.models:
            print(f"\nProcessing model: {model_spec.name}")

            model_result = {"model_name": model_spec.name, "assessments": []}

            # Process each assessment (AssessmentSpec object) for the current model
            for assessment_spec in self.evaluation_config.assessments:
                assessment_start_time = perf_counter()
                assessment_config = self.load_assessment_config(assessment_spec.config)

                # Execute inference for model + assessment pairing, with optional overrides
                inference_results = self.execute_inference_for_assessment(
                    model_spec=model_spec,
                    assessment_spec=assessment_spec,
                    output_dir=Path(output_dir) / model_spec.name / assessment_config.name,
                )
                # Process results and calculate metrics
                assessment_result = self.process_assessment_results(
                    assessment_config,
                    inference_results,
                    start_time=assessment_start_time,
                )

                model_result["assessments"].append(assessment_result)

            evaluation_report["model_results"].append(model_result)

        return evaluation_report

    def execute_metrics_only_pipeline(self, output_dir):
        """
        Execute metrics-only pipeline on existing inference results.
        """
        print("Starting metrics-only evaluation pipeline...")

        # Load inference results from configuration
        loaded_results = self._load_inference_results_from_config()

        # Group inference results by model only
        grouped_results = self._group_inference_results_by_model(loaded_results)

        evaluation_report = self._create_evaluation_report_structure(
            total_models=len(grouped_results),
            total_inference_files=len(loaded_results),
        )

        # Process each model's results
        for model_name, model_results in grouped_results.items():
            print(f"\nProcessing metrics for model: {model_name}")

            model_result = self._process_model_results_for_metrics_only(
                model_name, model_results
            )
            evaluation_report["model_results"].append(model_result)

        return evaluation_report

    # ------------------------------------------- #
    #                  Processing                 #
    # ------------------------------------------- #

    def execute_inference_for_assessment(self, model_spec, assessment_spec, output_dir):
        """
        Execute inference for a specific model and assessment using the inference engine.

        Args:
            model_spec: ModelSpec object with name, optional hyperparameters, optional quantization_config
            assessment_spec: AssessmentSpec object with config filename and optional hyperparameters
            output_dir: Output directory for results

        Returns: 
            List of inference result file paths.
        """
        # Load asssesment config
        assessment_config = self.load_assessment_config(assessment_spec.config)

        print(
            f"Executing inference for {model_spec.name} on assessment {assessment_config.name}"
        )

        # Merge hyperparameters: global < assessment < model
        global_hyperparams = self.evaluation_config.hyperparameters or {}
        assessment_hyperparams = assessment_spec.hyperparameters or {}
        model_hyperparams = model_spec.hyperparameters or {}

        merged_hyperparams = {
            **global_hyperparams,
            **assessment_hyperparams,
            **model_hyperparams
        }

        # Create inference engine instance
        logdir = Path(output_dir) / "inference_logs"
        inference_engine = Inference_Engine(logdir=str(logdir), promptdir=self.promptdir)

        environment_config = inference_engine.load_env_from_file(self.evaluation_config.environment_config)
        for avail_model in environment_config.models:

            if (
                "model_family" not in avail_model.keys()
                or "model_name" not in avail_model.keys()
            ):
                error_message = f"model_name and model_family are required for each model"
                self.raise_exception(error_message)

            klass = avail_model["model_family"]
            module = import_module(
                 "elm.inference_engine.languagemodels." + klass
            )
            available_model = module.Model(avail_model)
            inference_engine.available_models[avail_model["model_name"]] = available_model

        # Set model selection with merged hyperparameters
        inference_engine.model_selection = [{
            "name": model_spec.name,
            "hyperparameters": merged_hyperparams,
            "quantization_config": model_spec.quantization_config
        }]

        # Load prompts from assessment config
        for prompt_file in assessment_config.prompts:
            print(f"Loading prompts from: {prompt_file}")
            loaded_prompts = inference_engine.load_prompts_from_file(
                prompt_file, exit_on_fail=True
            )

            for prompt_config in loaded_prompts:
                inference_engine.available_prompts[prompt_config.name] = prompt_config
                inference_engine.prompt_selection.append(prompt_config.name)

        # Execute inference pipeline and collect list of result file paths
        results_dir = Path(output_dir) / "inference_results"
        result_files = inference_engine.execute_inference_pipeline(str(results_dir))

        print(f"Inference complete. Generated {len(result_files)} result files.")
        return result_files

    def process_assessment_results(self, assessment_config, inference_results, start_time=None):
        """
        Process inference results for a single assessment.

        Args:
            assessment_config: AssessmentConfig Pydantic object
            inference_results: List of result file paths
            start_time: Optional start time for execution time calculation
        
        Returns:
            Assessment result structure for evaluation report
        """
        # Load inference results into memory
        loaded_results = self.load_inference_results(inference_results)

        # Calculate metrics for this assessment
        metric_results = self.calculate_metrics(assessment_config, loaded_results)

        # Build prompt results by combining data from all metrics
        prompt_results = self._build_prompt_results(loaded_results, metric_results)

        execution_time = perf_counter() - start_time if start_time else None

        # Build assessment result structure
        assessment_result = {
            "name": assessment_config.name,
            "config": getattr(assessment_config, 'config_path', 'unknown'),
            "execution_time": round(execution_time, 1),
            "total_prompts": len(prompt_results),
            "metric_summaries": {
                name: result.get("summary", {})
                for name, result in metric_results.items()
            },
            "prompt_results": prompt_results,
        }

        return assessment_result

    def calculate_metrics(self, assessment_config, inference_results):
        """
        Calculate all metrics for an assessment.

        Args:
            assessment_config: AssessmentConfig Pydantic object
            inference_results: Loaded inference results
        
        Returns:
            Dictionary of metric results
        """
        print(f"Calculating metrics for assessment: {assessment_config.name}")

        metric_results = {}

        for metric_name in assessment_config.metrics:
            if metric_name not in self.available_metrics:
                print(f"Warning: Metric '{metric_name}' not available. Skipping.")
                continue

            print(f"Calculating {metric_name}...")

            try:
                metric = self.available_metrics[metric_name]
                result = metric.compute(inference_results)
                metric_results[metric_name] = result
                print(f"{metric_name}: completed")

            except Exception as e:
                print(f"Error calculating {metric_name}: {e}")
                # Continue with other metrics
                metric_results[metric_name] = {
                    "summary": {
                        "counts": {"total_items": 0, "scored_items": 0, "skipped_items": 0},
                        "scores": {},
                        "issues": [{"index": -1, "type": "calculation_error", "description": str(e)}]
                    },
                    "individual_results": [],
                }

        return metric_results

    def _load_inference_results_from_config(self):
        """
        Load inference results based on the configuration format.
        Returns loaded inference results.
        """
        inference_results_config = self.evaluation_config.inference_results

        if isinstance(inference_results_config, list):
            # Direct file path array format
            print(f"Loading {len(inference_results_config)} inference result files...")
            return self.load_inference_results(inference_results_config)
        elif isinstance(inference_results_config, InferenceResultsConfig):
            # InferenceResultsConfig object (only from_evaluation_report supported)
            if inference_results_config.type == "from_evaluation_report":
                return self._load_inference_results_from_evaluation_report(inference_results_config)
            else:
                raise ValueError(f"Unsupported inference_results type: {inference_results_config.type}")
        else:
            raise ValueError(f"Unsupported inference_results format: {type(inference_results_config)}")

    def load_inference_results(self, result_files):
        """Load inference result files into memory"""
        inference_results = []

        for file_path in result_files:
            try:
                with open(file_path, "r") as file:
                    result_data = json.load(file)
                    result_data["_source_file"] = file_path
                    inference_results.append(result_data)
            except Exception as e:
                print(f"Error loading inference result {file_path}: {e}")

        return inference_results

    def _load_inference_results_from_evaluation_report(self, config):
        """
        Load inference results from an evaluation report with optional filtering.

        Args:
            config: InferenceResultsConfig object with report_path and optional filter
        """
        report_path = os.path.expandvars(os.path.expanduser(config.report_path))

        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load evaluation report {report_path}: {e}")

        # Get filter criteria
        allowed_models = None
        allowed_assessments = None
        if config.filter:
            allowed_models = config.filter.models
            allowed_assessments = config.filter.assessments
        
        # Extract inference file paths from evaluation report
        file_paths = []
        
        for model_result in report.get("model_results", []):
            model_name = model_result.get("model_name", "unknown")
            # Apply model filter if specified
            if allowed_models and model_name not in allowed_models:
                continue
                
            for assessment_result in model_result.get("assessments", []):
                assessment_name = assessment_result.get("name", "unknown")
                # Apply assessment filter if specified
                if allowed_assessments and assessment_name not in allowed_assessments:
                    continue
                    
                for prompt_result in assessment_result.get("prompt_results", []):
                    source_file = prompt_result.get("source_file")
                    if source_file:
                        file_paths.append(source_file)
        
        if not file_paths:
            raise ValueError("No inference files found matching the specified filters")
        
        print(f"Loading {len(file_paths)} inference result files from evaluation report...")
        return self.load_inference_results(file_paths)

    def _group_inference_results_by_model(self, loaded_results):
        """
        Group inference results by model name only.
        Returns dict: {model_name: [results]}
        """
        grouped = {}

        for result in loaded_results:
            model_name = result.get("model_name", "unknown_model")

            if model_name not in grouped:
                grouped[model_name] = []

            grouped[model_name].append(result)

        return grouped

    def _process_model_results_for_metrics_only(self, model_name, model_results):
        """
        Process inference results for a single model in metrics_only pipeline.
        """
        start_time = perf_counter()

        # Get metrics to calculate from evaluation config
        metrics_to_calculate = self.evaluation_config.metrics or []
        if not metrics_to_calculate:
            raise ValueError("No metrics specified in evaluation config for metrics_only pipeline")

        # Calculate metrics for all results from this model
        metric_results = self._calculate_metrics_for_model(metrics_to_calculate, model_results)

        # Build prompt results with metric details
        prompt_results = self._build_prompt_results(model_results, metric_results)

        execution_time = perf_counter() - start_time

        return {
            "model_name": model_name,
            "total_inference_files": len(model_results),
            "execution_time": round(execution_time, 1),
            "metric_summaries": {
                name: result.get("summary", {})
                for name, result in metric_results.items()
            },
            "prompt_results": prompt_results,
        }

    def _calculate_metrics_for_model(self, metrics_to_calculate, model_results):
        """
        Calculate specified metrics for a model's inference results.
        """
        print(f"Calculating metrics: {metrics_to_calculate}")

        metric_results = {}

        for metric_name in metrics_to_calculate:
            if metric_name not in self.available_metrics:
                print(f"Warning: Metric '{metric_name}' not available. Skipping.")
                continue

            print(f"Calculating {metric_name}...")

            try:
                metric = self.available_metrics[metric_name]
                result = metric.compute(model_results)
                metric_results[metric_name] = result
                print(f"{metric_name}: completed")

            except Exception as e:
                print(f"Error calculating {metric_name}: {e}")
                metric_results[metric_name] = {
                    "summary": {
                        "counts": {"total_items": 0, "scored_items": 0, "skipped_items": 0},
                        "scores": {},
                        "issues": [{"index": -1, "type": "calculation_error", "description": str(e)}]
                    },
                    "individual_results": [],
                }

        return metric_results

    # ------------------------------------------- #
    #               Report Generation             #
    # ------------------------------------------- #

    def save_evaluation_report(self, evaluation_report, output_dir):
        """Save evaluation report to JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Use the run_id that was generated in the run() method
        run_id = getattr(self, "run_id", f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        report_path = output_path / f"evaluation_report_{run_id}.json"

        try:
            with open(report_path, "w") as file:
                json.dump(evaluation_report, file, indent=4)

            print(f"Evaluation report saved: {report_path}")
            return str(report_path)

        except Exception as e:
            print(f"Error saving evaluation report: {e}")
            raise

    # ------------------------------------------- #
    #               Helper Methods                #
    # ------------------------------------------- #

    def _create_evaluation_report_structure(self, total_models=0, total_assessments=0, **extra_metadata):
        """
        Create the base structure for evaluation reports.
        """
        metadata = {
            "run_id": getattr(self, "run_id", "unknown"),
            "evaluation_config": getattr(self, "config_file_path", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "pipeline_type": self.evaluation_config.pipeline_type,
            "total_models": total_models,
            "total_assessments": total_assessments,
        }
        # Add any extra metadata fields
        metadata.update(extra_metadata)

        return {
            "evaluation_metadata": metadata,
            "model_results": [],
        }

    def _build_prompt_results(self, loaded_results, metric_results):
        prompt_results = []

        for i, result in enumerate(loaded_results):
            prompt_config = result.get("prompt_config", {})
            prompt_result = {
                "name": prompt_config.get("name", "unknown"),
                "model_output": result.get("inference_results", ""),
                "inference_time": result.get("inference_time", 0.0),
                "source_file": result.get("_source_file", "unknown"),
            }

            # Add ground truth fields if they exist
            if "gt_text" in prompt_config and prompt_config["gt_text"] is not None:
                prompt_result["gt_text"] = prompt_config["gt_text"]

            if "gt_file" in prompt_config and prompt_config["gt_file"] is not None:
                prompt_result["gt_file"] = prompt_config["gt_file"]

            # Collect prompt-level metric details
            # Simple index-matching approach as individual_results matches 1:1 with inference_results order
            metric_details = {}
            for metric_name, metric_result in metric_results.items():
                individual_results = metric_result.get("individual_results", [])
                if i < len(individual_results):
                    individual_result = individual_results[i].copy()
                    individual_result.pop("prompt_name", None) # Remove redundant field
                    metric_details[metric_name] = individual_result
                else:
                    # Fallback if lists don't match -- this should not happen
                    metric_details[metric_name] = {
                        "skipped": True,
                        "reason": "index_mismatch_failure"
                    }

            prompt_results.append({**prompt_result, "metric_details": metric_details})

        return prompt_results

    def _get_available_metrics(self):
        try:
            klass_list = metrics.__all__
            for klass in klass_list:
                try:
                    module = import_module("metrics." + klass)
                    metric = module.Metric()
                    self.available_metrics[metric.name] = metric
                except Exception as e:
                    print(f"Warning: Failed to load metric {klass}: {e}")
        except Exception as e:
            print(f"Warning: Could not enumerate metrics - {e}")


    def _setup_directories(self, evaluation_configs_dir, assessment_configs_dir, results_dir):
        # Get directory where engine is located
        engine_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set config and results directories relative to the engine directory using overrides or defaults
        self.evaluation_configs_dir = evaluation_configs_dir or os.path.join(engine_dir, "evaluation_configs")
        self.assessment_configs_dir = assessment_configs_dir or os.path.join(engine_dir, "assessment_configs")
        self.results_dir = results_dir or os.path.join(engine_dir, "evaluation_results")
        
        # Validate directories exist (create results dir if it doesn't exist)
        self._validate_directory(self.evaluation_configs_dir, "evaluation configs", create_if_missing=False)
        self._validate_directory(self.assessment_configs_dir, "assessment configs", create_if_missing=False)
        self._validate_directory(self.results_dir, "results", create_if_missing=True)

    def _validate_directory(self, directory_path, directory_type, create_if_missing=False):
        if not os.path.exists(directory_path):
            if create_if_missing:
                os.makedirs(directory_path, exist_ok=True)
                print(f"Created {directory_type} directory: {directory_path}")
            else:
                raise FileNotFoundError(
                    f"{directory_type} directory not found: {directory_path}\n"
                    f"Please create this directory or specify a different path."
                )
        elif not os.path.isdir(directory_path):
            raise NotADirectoryError(f"{directory_type} path is not a directory: {directory_path}")
        
    def _resolve_config_filepath(self, filename, base_directory):
        """
        Resolution order:
        1) Absolute path (verbatim)
        2) Relative to current working directory
        3) Relative to base_directory
            - If the first path segment equals base dir name, try stripping it
        """
        base = Path(base_directory).resolve(strict=False)
        p = Path(os.path.expandvars(os.path.expanduser(filename)))

        # 1) Absolute → use as-is
        if p.is_absolute():
            q = p.resolve(strict=False)
            if q.exists() and q.is_file():
                return str(q)
            raise FileNotFoundError(f"Config file not found: {q}")

        candidates = []
        # 2) As-typed, relative to CWD
        candidates.append(Path.cwd() / p)

        # 3a) If user redundantly prefixed base dir name, try base/stripped
        if p.parts and p.parts[0] == base.name:
            stripped = Path(*p.parts[1:]) if len(p.parts) > 1 else Path("")
            if str(stripped):  # avoid empty
                candidates.append(base / stripped)

        # 3b) Finally, base/relative
        candidates.append(base / p)

        # Normalize, test, and collect tried list for good errors
        tried = []
        for cand in candidates:
            q = cand.resolve(strict=False)
            tried.append(q)
            if q.exists():
                return str(q)

        tried_list = "\n  - " + "\n  - ".join(str(t) for t in tried)
        raise FileNotFoundError(f"Config file not found. Tried:{tried_list}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute LLM evaluations using assessments and metrics."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to evaluation config JSON file (relative to evaluation_configs_dir).",
    )
    parser.add_argument(
        "--evaluation-configs-dir",
        type=str,
        help="Directory containing evaluation config files (default: engine_dir/evaluation_configs)",
    )
    parser.add_argument(
        "--assessment-configs-dir", 
        type=str,
        help="Directory containing assessment config files (default: engine_dir/assessment_configs)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory for evaluation results (default: engine_dir/evaluation_results)",
    )

    args = parser.parse_args()

    engine = Evaluation_Engine(
        evaluation_configs_dir=args.evaluation_configs_dir,
        assessment_configs_dir=args.assessment_configs_dir,
        results_dir=args.results_dir
    )
    engine.run(config_file=args.config)
