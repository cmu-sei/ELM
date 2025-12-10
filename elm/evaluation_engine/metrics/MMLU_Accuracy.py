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

from .MetricBase import MetricBase


class MMLU_Accuracy(MetricBase):
    
    @property 
    def name(self):
        return "MMLU_Accuracy"
    
    def compute(self, inference_results, assessment_config=None):
        """
        Compute accuracy across inference results.
        
        Args:
            inference_results: List of inference result dictionaries
            
        Returns:
            dict: Contains summary and individual results
        """
        total_items = len(inference_results) if inference_results else 0
        
        if total_items == 0:
            return self._empty_result()
        
        individual_results = []
        issues = []
        correct_count = 0
        skipped_count = 0
        failed_count = 0
        
        for i, result in enumerate(inference_results):
            prompt_name = result.get("prompt_config", {}).get("name", "unknown")
            model_output = result.get("inference_results", "")
            ground_truth = result.get("prompt_config", {}).get("gt_text")
            
            # Check for validation issues
            skip_reason = self._validate_inputs(model_output, ground_truth)
            
            if skip_reason:
                # Add skipped entry
                individual_results.append({
                    "prompt_name": prompt_name,
                    "status": "skipped",
                    "errors": [skip_reason]
                })
                issues.append({
                    "index": i,
                    "type": skip_reason,
                    "description": f"Skipped prompt {prompt_name}: {skip_reason}"
                })
                skipped_count += 1
            else:
                # Valid result - calculate scores
                try:
                    is_correct = self._check_correctness(model_output, ground_truth)
                    if is_correct:
                        correct_count += 1
                    
                    individual_results.append({
                        "prompt_name": prompt_name,
                        "status": "ok",
                        "errors": [],
                        "correct": is_correct
                    })

                except Exception as e:
                    individual_results.append({
                        "prompt_name": prompt_name,
                        "status": "failed",
                        "errors": [f"calculation_error: {str(e)}"]
                    })
                    issues.append({
                        "index": i,
                        "type": "calculation_error",
                        "description": f"MMLU_Accuracy calculation failed for prompt {prompt_name}: {str(e)}"
                    })
                    failed_count += 1
        
        scored_items = total_items - skipped_count - failed_count
        accuracy = correct_count / scored_items if scored_items > 0 else 0.0
        
        return {
            "summary": {
                "counts": {
                    "total_items": total_items,
                    "scored_items": scored_items,
                    "skipped_items": skipped_count,
                    "failed_items": failed_count,
                    "correct_answers": correct_count,
                    "incorrect_answers": scored_items - correct_count
                },
                "scores": {
                    "accuracy": accuracy,
                    "accuracy_percentage": accuracy * 100
                },
                "issues": issues,
            },
            "individual_results": individual_results
        }
    
    def _check_correctness(self, model_output, ground_truth):
        """
        Determine if model output matches ground truth.
                
        Returns:
            bool: True if model output matches ground truth, False otherwise
        """
        # Normalize both strings for comparison
        model_output_clean = str(model_output).strip().upper()
        ground_truth_clean = str(ground_truth).strip().upper()
        
        # Exact match
        if model_output_clean == ground_truth_clean:
            return True

        # For MMLU-style questions, extract first letter if model gives longer response
        if len(ground_truth_clean) == 1 and ground_truth_clean.isalpha():
            # Extract first letter from model output
            for char in model_output_clean:
                if char.isalpha():
                    return char.upper() == ground_truth_clean
        
        return False
    
    def _validate_inputs(self, model_output, ground_truth):
        if not model_output or not str(model_output).strip():
            return "empty_model_output"
        if ground_truth is None or not str(ground_truth).strip():
            return "empty_ground_truth" 
        return None

    def _empty_result(self):
        return {
            "summary": {
                "counts": {"total_items": 0, "scored_items": 0, "skipped_items": 0},
                "scores": {},
                "issues": []
            },
            "individual_results": []
        }

# Factory function for metric instantiation
def Metric():
    return MMLU_Accuracy()