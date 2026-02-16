"""Core modules for Veritas research agent."""

from veritas.core.agent import ResearchAgent, ResearchTask, ResearchResult
from veritas.core.browser import BrowserSession
from veritas.core.uncertainty import (
    UncertaintyQuantifier,
    VoteSampler,
    normalize_uncertainty,
    uncertainty_to_confidence,
)

__all__ = [
    "ResearchAgent",
    "ResearchTask",
    "ResearchResult",
    "BrowserSession",
    "UncertaintyQuantifier",
    "VoteSampler",
    "normalize_uncertainty",
    "uncertainty_to_confidence",
]
