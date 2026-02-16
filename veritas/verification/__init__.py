"""Verification module implementing CM2 checklist-based rewards."""

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
    CHECKLIST_TEMPLATES,
)

__all__ = [
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
    "CHECKLIST_TEMPLATES",
]
