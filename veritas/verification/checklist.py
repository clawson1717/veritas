"""CM2-inspired checklist-based verification system.

Implements binary checklist rewards for multi-step agent verification,
providing fine-grained quality assessment without requiring verifiable
outcome rewards.

From the CM2 paper: "CM2: Reinforcement Learning with Checklist Rewards 
for Multi-Turn and Multi-Step Agentic Tool Use"

Key insight: Binary criteria are easier to evaluate than full outcomes,
and provide more granular reward signals for training multi-step agents.
"""

from typing import Callable, Any, Protocol
from dataclasses import dataclass, field
from enum import Enum
import re


class CheckStatus(Enum):
    """Status of a checklist item."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Context(Protocol):
    """Protocol for verification context."""
    content: str
    metadata: dict[str, Any]


@dataclass
class ChecklistItem:
    """A single checklist verification item with binary pass/fail criteria.
    
    Attributes:
        id: Unique identifier for this check
        description: Human-readable description of what is being checked
        check_fn: Function that evaluates the context and returns bool
        status: Current status of this check
        required: Whether this check must pass for overall success
        category: Optional category for grouping related checks
        weight: Weight for scoring (default 1.0)
    """
    id: str
    description: str
    check_fn: Callable[[Any], bool]
    status: CheckStatus = field(default=CheckStatus.PENDING)
    required: bool = field(default=True)
    category: str = field(default="general")
    weight: float = field(default=1.0)
    
    def evaluate(self, context: Any) -> bool:
        """Evaluate this checklist item against context.
        
        Args:
            context: The data to verify (step output, retrieval result, etc.)
            
        Returns:
            True if check passes, False otherwise
        """
        try:
            result = self.check_fn(context)
            self.status = CheckStatus.PASSED if result else CheckStatus.FAILED
            return result
        except Exception as e:
            # Log error but don't crash
            self.status = CheckStatus.FAILED
            return False
    
    def reset(self) -> None:
        """Reset status to pending."""
        self.status = CheckStatus.PENDING


@dataclass
class ChecklistResult:
    """Result of a checklist evaluation.
    
    Attributes:
        total: Total number of checks
        passed: Number of checks that passed
        failed: Number of checks that failed
        skipped: Number of checks that were skipped
        required_failed: Number of required checks that failed
        all_passed: Whether all checks passed
        score: Weighted score (0.0 to 1.0)
        by_category: Breakdown of results by category
    """
    total: int
    passed: int
    failed: int
    skipped: int
    required_failed: int
    all_passed: bool
    score: float
    by_category: dict[str, dict[str, int]] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "required_failed": self.required_failed,
            "all_passed": self.all_passed,
            "score": round(self.score, 3),
            "by_category": self.by_category,
        }


class ChecklistVerifier:
    """Binary checklist verification for agent steps.
    
    Inspired by CM2 (Checklist-guided Multi-step Multi-turn),
    this verifier provides:
    - Fine-grained binary criteria per step
    - No need for verifiable end-to-end outcomes
    - Reward signal for RL training
    - Structured feedback for agent improvement
    - Category-based organization
    
    Example:
        >>> verifier = ChecklistVerifier([
        ...     ChecklistItem("has_url", "Result contains a URL", 
        ...                   lambda x: "http" in str(x)),
        ...     ChecklistItem("not_empty", "Result is not empty", 
        ...                   lambda x: len(str(x)) > 0),
        ... ])
        >>> result = verifier.verify("Visit https://example.com")
        >>> print(result.score)
        1.0
    """
    
    def __init__(self, items: list[ChecklistItem] | None = None, name: str = "default"):
        """Initialize verifier with checklist items.
        
        Args:
            items: List of checklist items to verify
            name: Name of this verifier (for tracking)
        """
        self.items = items or []
        self.name = name
        self._history: list[ChecklistResult] = []
    
    def add_item(self, item: ChecklistItem) -> None:
        """Add a checklist item."""
        self.items.append(item)
    
    def add_items(self, items: list[ChecklistItem]) -> None:
        """Add multiple checklist items."""
        self.items.extend(items)
    
    def verify(self, context: Any, reset: bool = True) -> ChecklistResult:
        """Run all checklist items against the context.
        
        Args:
            context: The data to verify (step output, retrieval result, etc.)
            reset: Whether to reset item statuses before evaluation
            
        Returns:
            ChecklistResult with pass/fail statistics
        """
        if reset:
            for item in self.items:
                item.reset()
        
        passed = 0
        failed = 0
        skipped = 0
        required_failed = 0
        total_weight = 0.0
        passed_weight = 0.0
        
        # Track by category
        by_category: dict[str, dict[str, int]] = {}
        
        for item in self.items:
            if item.status == CheckStatus.SKIPPED:
                skipped += 1
                continue
            
            if item.evaluate(context):
                passed += 1
                passed_weight += item.weight
            else:
                failed += 1
                if item.required:
                    required_failed += 1
            
            total_weight += item.weight
            
            # Track by category
            if item.category not in by_category:
                by_category[item.category] = {"passed": 0, "failed": 0, "total": 0}
            by_category[item.category]["total"] += 1
            if item.status == CheckStatus.PASSED:
                by_category[item.category]["passed"] += 1
            else:
                by_category[item.category]["failed"] += 1
        
        total = len(self.items)
        all_passed = failed == 0 and total > 0
        score = passed_weight / total_weight if total_weight > 0 else 0.0
        
        result = ChecklistResult(
            total=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            required_failed=required_failed,
            all_passed=all_passed,
            score=score,
            by_category=by_category,
        )
        self._history.append(result)
        return result
    
    def get_reward(self, result: ChecklistResult | None = None) -> float:
        """Calculate reward signal for RL training.
        
        CM2-style reward calculation:
        - +1.0 for all checks passing
        - Penalty proportional to required failures
        - Partial credit for optional failures
        
        Args:
            result: Checklist result to calculate reward from
            
        Returns:
            Reward value (-1.0 to 1.0)
        """
        if result is None:
            if not self._history:
                return 0.0
            result = self._history[-1]
        
        if result.all_passed:
            return 1.0
        elif result.required_failed > 0:
            # Penalty for required failures, but not total disaster
            return max(-1.0, -0.3 * result.required_failed)
        else:
            # Some optional checks failed, partial credit
            return 0.3 + (result.score * 0.4)
    
    def get_feedback(self, result: ChecklistResult | None = None) -> str:
        """Generate human-readable feedback from verification.
        
        Args:
            result: Checklist result to generate feedback from
            
        Returns:
            Formatted feedback string
        """
        if result is None:
            if not self._history:
                return "No verification performed yet."
            result = self._history[-1]
        
        lines = [f"Verification: {self.name}", "=" * 40]
        
        for item in self.items:
            status_icon = "✓" if item.status == CheckStatus.PASSED else "✗" if item.status == CheckStatus.FAILED else "○"
            required_marker = " (required)" if item.required else ""
            lines.append(f"  {status_icon} {item.description}{required_marker}")
        
        lines.append("-" * 40)
        lines.append(f"Score: {result.score:.1%} ({result.passed}/{result.total})")
        
        if result.all_passed:
            lines.append("Status: All checks passed ✓")
        elif result.required_failed > 0:
            lines.append(f"Status: {result.required_failed} required check(s) failed")
        else:
            lines.append("Status: Some optional checks failed")
        
        return "\n".join(lines)
    
    def get_history(self) -> list[ChecklistResult]:
        """Get verification history."""
        return self._history.copy()
    
    def reset(self) -> None:
        """Reset all items and clear history."""
        for item in self.items:
            item.reset()
        self._history.clear()


# =============================================================================
# Predefined Checklists for Common Verification Scenarios
# =============================================================================

def create_retrieval_checklist() -> ChecklistVerifier:
    """Create a checklist for verifying web retrieval results.
    
    Checks:
    - Content is not empty
    - Content has minimum length (not just error pages)
    - Contains actual text (not just nav/footer)
    - Has reasonable length (not truncated)
    """
    items = [
        ChecklistItem(
            id="not_empty",
            description="Retrieved content is not empty",
            check_fn=lambda x: bool(str(x).strip()),
            category="basic",
        ),
        ChecklistItem(
            id="min_length",
            description="Content has meaningful length (>100 chars)",
            check_fn=lambda x: len(str(x)) > 100,
            category="basic",
        ),
        ChecklistItem(
            id="has_substance",
            description="Content contains substantive text (not just nav)",
            check_fn=lambda x: len(re.findall(r'[a-zA-Z]{4,}', str(x))) > 5,
            category="quality",
        ),
        ChecklistItem(
            id="not_error_page",
            description="Not an error page (no 404/500 messages)",
            check_fn=lambda x: not any(err in str(x).lower() for err in [
                "404 not found", "500 internal server", "error occurred",
                "page not found", "access denied"
            ]),
            category="basic",
            required=True,
        ),
    ]
    return ChecklistVerifier(items, name="retrieval")


def create_synthesis_checklist() -> ChecklistVerifier:
    """Create a checklist for verifying answer synthesis results.
    
    Checks:
    - Answer is not empty
    - Answer directly addresses the question
    - Answer cites sources if applicable
    - Answer has appropriate length
    """
    items = [
        ChecklistItem(
            id="not_empty",
            description="Answer is not empty",
            check_fn=lambda x: bool(str(x).strip()),
            category="basic",
            required=True,
        ),
        ChecklistItem(
            id="has_substance",
            description="Answer has substantive content (>50 chars)",
            check_fn=lambda x: len(str(x)) > 50,
            category="basic",
        ),
        ChecklistItem(
            id="addresses_question",
            description="Answer appears to address the original question",
            check_fn=lambda x: not any(phrase in str(x).lower() for phrase in [
                "i cannot", "i can't", "i don't know", "no information",
                "unable to", "not found"
            ]),
            category="relevance",
            required=True,
        ),
        ChecklistItem(
            id="proper_length",
            description="Answer is not excessively long (<5000 chars)",
            check_fn=lambda x: len(str(x)) < 5000,
            category="quality",
            required=False,
        ),
        ChecklistItem(
            id="has_structure",
            description="Answer has some structure (paragraphs or bullets)",
            check_fn=lambda x: '\n' in str(x) or '•' in str(x) or '-' in str(x),
            category="quality",
            required=False,
        ),
    ]
    return ChecklistVerifier(items, name="synthesis")


def create_search_checklist() -> ChecklistVerifier:
    """Create a checklist for verifying search results.
    
    Checks:
    - Results were found
    - Results contain URLs
    - Minimum number of results
    """
    items = [
        ChecklistItem(
            id="has_results",
            description="Search returned results",
            check_fn=lambda x: bool(x) and (isinstance(x, list) and len(x) > 0 or isinstance(x, str) and len(x) > 0),
            category="basic",
            required=True,
        ),
        ChecklistItem(
            id="min_results",
            description="At least 3 results found",
            check_fn=lambda x: len(x) >= 3 if isinstance(x, list) else True,
            category="coverage",
            required=False,
        ),
        ChecklistItem(
            id="has_urls",
            description="Results contain URLs",
            check_fn=lambda x: "http" in str(x).lower(),
            category="basic",
        ),
    ]
    return ChecklistVerifier(items, name="search")


def create_research_quality_checklist() -> ChecklistVerifier:
    """Create a comprehensive checklist for end-to-end research quality.
    
    Checks:
    - Multiple sources consulted
    - Answer is comprehensive
    - No obvious hallucinations
    - Appropriate uncertainty expressed
    """
    items = [
        ChecklistItem(
            id="multiple_sources",
            description="Research consulted multiple sources",
            check_fn=lambda ctx: len(ctx.get("sources", [])) >= 2 if isinstance(ctx, dict) else False,
            category="coverage",
        ),
        ChecklistItem(
            id="comprehensive_answer",
            description="Answer is comprehensive (>200 chars)",
            check_fn=lambda ctx: len(str(ctx.get("answer", ""))) > 200 if isinstance(ctx, dict) else len(str(ctx)) > 200,
            category="quality",
        ),
        ChecklistItem(
            id="no_hallucination_markers",
            description="No obvious hallucination markers",
            check_fn=lambda ctx: not any(phrase in str(ctx).lower() for phrase in [
                "as an ai", "my training data", "i don't have access",
                "i cannot browse", "cutoff knowledge"
            ]),
            category="accuracy",
            required=True,
        ),
        ChecklistItem(
            id="clear_confidence",
            description="Answer expresses appropriate confidence",
            check_fn=lambda ctx: any(phrase in str(ctx).lower() for phrase in [
                "according to", "sources indicate", "research shows",
                "based on", "found that"
            ]) or "highly confident" in str(ctx).lower(),
            category="transparency",
            required=False,
        ),
    ]
    return ChecklistVerifier(items, name="research_quality")


# Registry of predefined checklists
CHECKLIST_TEMPLATES = {
    "retrieval": create_retrieval_checklist,
    "synthesis": create_synthesis_checklist,
    "search": create_search_checklist,
    "research_quality": create_research_quality_checklist,
}


def get_checklist(name: str) -> ChecklistVerifier:
    """Get a predefined checklist by name.
    
    Args:
        name: Name of the checklist template
        
    Returns:
        Configured ChecklistVerifier
        
    Raises:
        ValueError: If checklist name is unknown
    """
    if name not in CHECKLIST_TEMPLATES:
        available = ", ".join(CHECKLIST_TEMPLATES.keys())
        raise ValueError(f"Unknown checklist '{name}'. Available: {available}")
    
    return CHECKLIST_TEMPLATES[name]()


def list_checklists() -> list[str]:
    """List available checklist template names."""
    return list(CHECKLIST_TEMPLATES.keys())
