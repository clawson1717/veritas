"""Veritas - An adaptive web research agent.

Veritas is a research agent that combines:
- CATTS: Dynamic compute allocation via uncertainty
- CM2: Checklist-based verification with binary criteria
- iGRPO: Self-feedback-driven iterative refinement
- LawThinker: Verification after every retrieval
- DyTopo: Dynamic topology via semantic matching
"""

__version__ = "0.1.0"
__author__ = "Veritas Team"

from veritas.core.agent import ResearchAgent, ResearchTask, ResearchResult
from veritas.core.browser import BrowserSession
from veritas.core.uncertainty import (
    UncertaintyQuantifier,
    VoteSampler,
    normalize_uncertainty,
    uncertainty_to_confidence,
)
from veritas.allocation.catts import (
    CATTSAllocator,
    AllocationConfig,
    AllocationDecision,
    AllocationStrategy,
    BudgetManager,
)
from veritas.verification.checklist import (
    ChecklistVerifier,
    ChecklistItem,
    ChecklistResult,
    CheckStatus,
    create_retrieval_checklist,
    create_synthesis_checklist,
    create_search_checklist,
    create_research_quality_checklist,
    get_checklist,
    list_checklists,
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
    "CATTSAllocator",
    "AllocationConfig",
    "AllocationDecision",
    "AllocationStrategy",
    "BudgetManager",
    "ChecklistVerifier",
    "ChecklistItem",
    "ChecklistResult",
    "CheckStatus",
    "create_retrieval_checklist",
    "create_synthesis_checklist",
    "create_search_checklist",
    "create_research_quality_checklist",
    "get_checklist",
    "list_checklists",
]
