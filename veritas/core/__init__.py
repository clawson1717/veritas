"""Core modules for Veritas research agent."""

from veritas.core.agent import ResearchAgent, ResearchTask, ResearchResult
from veritas.core.browser import BrowserSession
from veritas.core.uncertainty import (
    UncertaintyQuantifier,
    VoteSampler,
    normalize_uncertainty,
    uncertainty_to_confidence,
)
from veritas.core.integrator import (
    AgentIntegrator,
    IntegrationConfig,
    IntegrationResult,
    create_integrator,
    create_minimal_integrator,
    create_full_integrator,
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
    "AgentIntegrator",
    "IntegrationConfig",
    "IntegrationResult",
    "create_integrator",
    "create_minimal_integrator",
    "create_full_integrator",
]
