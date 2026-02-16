"""Veritas refinement module - iGRPO self-feedback refinement.

Provides iterative refinement capabilities where models learn to
improve their own outputs based on self-generated critiques.
"""

from veritas.refinement.igrpo import (
    RefinementResult,
    SelfFeedbackRefiner,
    ConvergenceChecker,
    ConvergenceConfig,
    SimilarityMetric,
    refine_response,
)

__all__ = [
    "RefinementResult",
    "SelfFeedbackRefiner",
    "ConvergenceChecker",
    "ConvergenceConfig",
    "SimilarityMetric",
    "refine_response",
]
