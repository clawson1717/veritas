"""CM2-inspired checklist-based verification system.

Implements binary checklist rewards for multi-step agent verification,
providing fine-grained quality assessment without requiring verifiable
outcome rewards.
"""

from typing import Callable
from dataclasses import dataclass, field
from enum import Enum


class CheckStatus(Enum):
    """Status of a checklist item."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class ChecklistItem:
    """A single checklist verification item."""
    id: str
    description: str
    check_fn: Callable[[any], bool]
    status: CheckStatus = CheckStatus.PENDING
    required: bool = True
    
    def evaluate(self, context: any) -> bool:
        """Evaluate this checklist item against context."""
        try:
            result = self.check_fn(context)
            self.status = CheckStatus.PASSED if result else CheckStatus.FAILED
            return result
        except Exception:
            self.status = CheckStatus.FAILED
            return False


@dataclass
class ChecklistResult:
    """Result of a checklist evaluation."""
    total: int
    passed: int
    failed: int
    required_failed: int
    all_passed: bool
    score: float  # 0.0 to 1.0


class ChecklistVerifier:
    """Binary checklist verification for agent steps.
    
    Inspired by CM2 (Checklist-guided Multi-step Multi-turn),
    this verifier provides:
    - Fine-grained binary criteria per step
    - No need for verifiable end-to-end outcomes
    - Reward signal for RL training
    - Structured feedback for agent improvement
    
    Example:
        verifier = ChecklistVerifier([
            ChecklistItem("has_url", "Result contains a URL", lambda x: "http" in x),
            ChecklistItem("not_empty", "Result is not empty", lambda x: len(x) > 0),
        ])
        result = verifier.verify(step_output)
    """
    
    def __init__(self, items: list[ChecklistItem] | None = None):
        self.items = items or []
        self._history: list[ChecklistResult] = []
    
    def add_item(self, item: ChecklistItem) -> None:
        """Add a checklist item."""
        self.items.append(item)
    
    def verify(self, context: any, reset: bool = True) -> ChecklistResult:
        """Run all checklist items against the context.
        
        Args:
            context: The data to verify (step output, retrieval result, etc.)
            reset: Whether to reset item statuses before evaluation
            
        Returns:
            ChecklistResult with pass/fail statistics
        """
        if reset:
            for item in self.items:
                item.status = CheckStatus.PENDING
        
        passed = 0
        failed = 0
        required_failed = 0
        
        for item in self.items:
            if item.evaluate(context):
                passed += 1
            else:
                failed += 1
                if item.required:
                    required_failed += 1
        
        total = len(self.items)
        all_passed = failed == 0 and total > 0
        score = passed / total if total > 0 else 0.0
        
        result = ChecklistResult(
            total=total,
            passed=passed,
            failed=failed,
            required_failed=required_failed,
            all_passed=all_passed,
            score=score,
        )
        self._history.append(result)
        return result
    
    def get_reward(self, result: ChecklistResult | None = None) -> float:
        """Calculate reward signal for RL training.
        
        Returns high reward for all checks passing,
        penalty for required checks failing.
        
        Args:
            result: Checklist result to calculate reward from
            
        Returns:
            Reward value (typically -1 to 1)
        """
        if result is None:
            if not self._history:
                return 0.0
            result = self._history[-1]
        
        if result.all_passed:
            return 1.0
        elif result.required_failed > 0:
            return -0.5 * result.required_failed
        else:
            return result.score * 0.5
