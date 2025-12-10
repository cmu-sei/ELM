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

from abc import ABC, abstractmethod

class MetricBase(ABC):
    
    @property
    @abstractmethod
    def name(self):
        """Unique identifier for this metric type."""
        pass
    
    @abstractmethod
    def compute(self):
        """
        Compute the metric across inference results.
        
        Args:
            inference_results: List of inference result dictionaries
            
        Returns:
            dict: Must contain 'summary' dict and 'individual_results' list
                {
                    "summary": {
                        "counts": {
                            "total_items": int,      # Total inference results provided
                            "scored_items": int,     # Successfully scored items
                            "skipped_items": int     # Items skipped due to issues
                        },
                        "scores": {
                            # Metric-specific scores (e.g., "accuracy": 0.85)
                        },
                        "issues": [
                            # List of problems encountered during scoring
                            {"index": int, "type": str, "description": str}
                        ],
                        # Additional metric-specific summary data
                    },
                    "individual_results": [
                        # List of per-prompt results (can be empty if not applicable)
                        # Each item should contain prompt-level analysis
                    ]
                }
        """
        pass