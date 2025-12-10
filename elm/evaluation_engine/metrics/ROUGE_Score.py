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
from rouge_score import rouge_scorer, scoring
import nltk


class ROUGE_Score(MetricBase):
    """
    Computes ROUGE scores between two blobs of text (model_output and ground_truth).

    Note that ROUGE is case insensitive.

    Requires nltk==3.9.1
    Requires nltk data `punkt` and `punk_tab`

    This metric is a wrapper around Google Research reimplementation of ROUGE:
    https://github.com/google-research/google-research/tree/master/rouge
    """
    
    def __init__(self):
        # Ensure NLTK data required for `split_summaries=True` is available
        self._ensure_nltk_punkt()

        # Use stemmer to improve matching by removing word suffixes
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            use_stemmer=True,
            split_summaries=True
        )

    
    @property 
    def name(self):
        return "ROUGE_Score"
    
    def compute(self, inference_results):
        """
        Compute ROUGE scores for inference results using rouge-score package.
        
        Args:
            inference_results: List of inference result dictionaries
            
        Returns:
            dict: Contains summary and individual results
        """
        total_items = len(inference_results) if inference_results else 0
        
        if total_items == 0:
            return self._empty_result()
        
        individual_results = []
        skipped_count = 0
        failed_count = 0
        issues = []
        aggregator = scoring.BootstrapAggregator()
        
        for i, result in enumerate(inference_results):
            prompt_name = result.get("prompt_config", {}).get("name", "unknown")
            model_output = result.get("inference_results", "")
            ground_truth = result.get("prompt_config", {}).get("gt_text", "")
            
            # Validate inputs
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
                    native_scores = self.scorer.score(str(ground_truth), str(model_output))
                    
                    individual_results.append({
                        "prompt_name": prompt_name,
                        "status": "ok",
                        "errors": [],
                        "rouge_scores": {
                            rouge_type: {
                                "precision": round(score.precision, 4),
                                "recall": round(score.recall, 4),
                                "fmeasure": round(score.fmeasure, 4)
                            }
                            for rouge_type, score in native_scores.items()
                        }
                    })   
                    # Add to aggregator for summary statistics
                    aggregator.add_scores(native_scores)
                    
                except Exception as e:
                    individual_results.append({
                        "prompt_name": prompt_name,
                        "status": "failed",
                        "errors": [f"calculation_error: {str(e)}"]
                    })
                    issues.append({
                        "index": i,
                        "type": "calculation_error",
                        "description": f"ROUGE calculation failed for prompt {prompt_name}: {str(e)}"
                    })
                    failed_count += 1

        scored_items = total_items - skipped_count - failed_count
        if scored_items > 0:
            # Get bootstrap aggregated scores per original ROUGE implementation
            aggregated = aggregator.aggregate()
            summary_scores = {
                rouge_type: {
                    "precision": round(agg_score.mid.precision, 4),
                    "recall": round(agg_score.mid.recall, 4), 
                    "fmeasure": round(agg_score.mid.fmeasure, 4),
                    "confidence_interval": {
                        "low": {
                            "precision": round(agg_score.low.precision, 4),
                            "recall": round(agg_score.low.recall, 4),
                            "fmeasure": round(agg_score.low.fmeasure, 4)
                        },
                        "high": {
                            "precision": round(agg_score.high.precision, 4),
                            "recall": round(agg_score.high.recall, 4),
                            "fmeasure": round(agg_score.high.fmeasure, 4)
                        }
                    }
                }
                for rouge_type, agg_score in aggregated.items()
            }
        else:
            summary_scores = {}
        
        return {
            "summary": {
                "counts": {
                    "total_items": total_items,
                    "scored_items": scored_items,
                    "skipped_items": skipped_count,
                    "failed_items": failed_count
                },
                "scores": summary_scores,
                "issues": issues
            },
            "individual_results": individual_results
        }
    
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
    
    def _ensure_nltk_punkt(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading required NLTK data (`punkt`) for ROUGE scoring...")
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Downloading required NLTK data (`punkt_tab`) for ROUGE scoring...")
            nltk.download('punkt_tab', quiet=True)
    
# Factory function for metric instantiation
def Metric():
    return ROUGE_Score()