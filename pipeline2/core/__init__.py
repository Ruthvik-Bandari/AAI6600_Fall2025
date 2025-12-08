"""
Pipeline 2 - Core Modules

Embedding-based mental health classification and facility matching system.
"""

__version__ = "1.0.0"
__author__ = "AAI6600 Fall 2025 - Northeastern University"

from .facility_scorer import FacilityScorer
from .anti_hallucination import AntiHallucinationValidator, NegativeEvidenceDetector
from .mental_health_classifier import MentalHealthClassifier

__all__ = [
    'FacilityScorer',
    'AntiHallucinationValidator',
    'NegativeEvidenceDetector',
    'MentalHealthClassifier'
]

