"""CATTS (Compute Allocation via Test-Time Scaling) allocator.

Dynamically allocates compute resources based on uncertainty statistics
derived from vote distributions, as described in the CATTS paper.

Key insight from CATTS: Allocate more compute to uncertain steps,
less to confident ones. Achieves better performance with fewer tokens
than uniform allocation.
"""

from typing import Callable, Any, Literal
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict

from veritas.core.uncertainty import UncertaintyQuantifier, uncertainty_to_confidence


class AllocationStrategy(Enum):
    """Strategy for compute allocation."""
    UNIFORM = "uniform"  # Baseline: same allocation every step
    UNCERTAINTY = "uncertainty"  # Scale with uncertainty
    ADAPTIVE = "adaptive"  # Consider step type and history


@dataclass
class AllocationConfig:
    """Configuration for CATTS compute allocation.
    
    Attributes:
        min_samples: Minimum number of reasoning samples
        max_samples: Maximum number of reasoning samples
        confidence_threshold: Confidence level to stop early
        token_budget_base: Base token budget for confident steps
        token_budget_max: Maximum token budget for uncertain steps
        strategy: Allocation strategy to use
        step_multipliers: Optional multipliers for different step types
    """
    min_samples: int = 3
    max_samples: int = 10
    confidence_threshold: float = 0.8
    token_budget_base: int = 1000
    token_budget_max: int = 5000
    strategy: AllocationStrategy = AllocationStrategy.ADAPTIVE
    step_multipliers: dict[str, float] = field(default_factory=lambda: {
        "search": 1.0,
        "synthesis": 1.2,
        "verification": 0.8,
        "refinement": 1.5,
    })


@dataclass
class AllocationDecision:
    """A compute allocation decision.
    
    Attributes:
        samples: Number of reasoning samples to generate
        token_budget: Token budget for this step
        uncertainty: Measured uncertainty (0-1)
        confidence: Confidence score (0-1)
        should_continue: Whether to continue sampling
        reasoning: Explanation for the allocation
    """
    samples: int
    token_budget: int
    uncertainty: float
    confidence: float
    should_continue: bool
    reasoning: str


class CATTSAllocator:
    """Dynamic compute allocator based on uncertainty quantification.
    
    Implements the CATTS approach:
    1. Sample multiple reasoning paths (votes)
    2. Calculate uncertainty from vote distribution
    3. Allocate more compute to high-uncertainty steps
    4. Stop early when confidence is sufficient
    
    Args:
        config: Allocation configuration parameters
        uncertainty_fn: Optional custom uncertainty calculator
    
    Example:
        >>> allocator = CATTSAllocator()
        >>> votes = ["answer A", "answer A", "answer B"]
        >>> decision = allocator.allocate(votes, step_type="synthesis")
        >>> print(decision.token_budget)
        3400
    """
    
    def __init__(
        self,
        config: AllocationConfig | None = None,
        uncertainty_fn: Callable[[list], float] | None = None,
    ):
        self.config = config or AllocationConfig()
        self._uncertainty_quantifier = UncertaintyQuantifier()
        self._uncertainty_fn = uncertainty_fn or self._default_uncertainty
        self._budget_history: list[dict] = []
        self._total_tokens_used = 0
    
    def allocate(
        self,
        vote_distribution: list[Any],
        step_type: str = "general",
        step_context: dict | None = None,
    ) -> AllocationDecision:
        """Allocate compute resources based on uncertainty.
        
        Args:
            vote_distribution: Samples/votes from multiple reasoning paths
            step_type: Type of step (search, synthesis, verification, etc.)
            step_context: Additional context about the current step
            
        Returns:
            AllocationDecision with budget, samples, and continuation flag
        """
        step_context = step_context or {}
        
        # Calculate uncertainty and confidence
        uncertainty = self._uncertainty_fn(vote_distribution)
        confidence = uncertainty_to_confidence(uncertainty)
        
        # Determine allocation based on strategy
        if self.config.strategy == AllocationStrategy.UNIFORM:
            samples, tokens, reasoning = self._uniform_allocation()
        elif self.config.strategy == AllocationStrategy.UNCERTAINTY:
            samples, tokens, reasoning = self._uncertainty_based_allocation(uncertainty, confidence)
        else:  # ADAPTIVE
            samples, tokens, reasoning = self._adaptive_allocation(
                uncertainty, confidence, step_type, step_context
            )
        
        # Determine if we should continue sampling
        should_continue = confidence < self.config.confidence_threshold and samples < self.config.max_samples
        
        # Track budget usage
        self._budget_history.append({
            "step_type": step_type,
            "uncertainty": uncertainty,
            "confidence": confidence,
            "tokens": tokens,
            "samples": samples,
        })
        self._total_tokens_used += tokens
        
        return AllocationDecision(
            samples=samples,
            token_budget=tokens,
            uncertainty=uncertainty,
            confidence=confidence,
            should_continue=should_continue,
            reasoning=reasoning,
        )
    
    def allocate_for_generation(
        self,
        prompt: str,
        base_tokens: int = 1000,
        min_samples: int = 3,
    ) -> dict:
        """Pre-allocate before generating votes (when votes don't exist yet).
        
        Uses heuristics based on prompt characteristics to estimate
        appropriate initial allocation.
        
        Args:
            prompt: The prompt that will be sent to the LLM
            base_tokens: Base token estimate
            min_samples: Minimum samples to start with
            
        Returns:
            Initial allocation parameters
        """
        # Estimate complexity from prompt
        complexity_signals = [
            len(prompt) > 500,  # Long prompt
            "?" in prompt,  # Question
            "compare" in prompt.lower(),  # Comparison task
            "analyze" in prompt.lower(),  # Analysis task
            "explain" in prompt.lower(),  # Explanation task
            "why" in prompt.lower(),  # Causal reasoning
        ]
        
        complexity = sum(complexity_signals) / len(complexity_signals)
        
        # Scale initial allocation
        tokens = int(base_tokens * (1 + complexity))
        samples = min(min_samples + int(complexity * 2), self.config.max_samples)
        
        return {
            "token_budget": min(tokens, self.config.token_budget_max),
            "samples": samples,
            "estimated_complexity": complexity,
        }
    
    def get_budget_summary(self) -> dict:
        """Get summary of budget usage across all allocations.
        
        Returns:
            Dictionary with total usage, averages, and step breakdown
        """
        if not self._budget_history:
            return {"total_tokens": 0, "steps": 0}
        
        total_tokens = sum(h["tokens"] for h in self._budget_history)
        avg_uncertainty = sum(h["uncertainty"] for h in self._budget_history) / len(self._budget_history)
        
        # Breakdown by step type
        by_step = defaultdict(lambda: {"tokens": 0, "count": 0})
        for h in self._budget_history:
            by_step[h["step_type"]]["tokens"] += h["tokens"]
            by_step[h["step_type"]]["count"] += 1
        
        return {
            "total_tokens": total_tokens,
            "steps": len(self._budget_history),
            "avg_uncertainty": avg_uncertainty,
            "by_step_type": dict(by_step),
            "savings_vs_uniform": self._calculate_savings(),
        }
    
    def reset(self) -> None:
        """Reset budget tracking history."""
        self._budget_history = []
        self._total_tokens_used = 0
    
    def _uniform_allocation(self) -> tuple[int, int, str]:
        """Baseline uniform allocation (no uncertainty scaling)."""
        return (
            self.config.min_samples,
            self.config.token_budget_base,
            "Uniform allocation (baseline)",
        )
    
    def _uncertainty_based_allocation(
        self,
        uncertainty: float,
        confidence: float,
    ) -> tuple[int, int, str]:
        """Allocate purely based on uncertainty."""
        if confidence >= self.config.confidence_threshold:
            return (
                self.config.min_samples,
                self.config.token_budget_base,
                f"High confidence ({confidence:.2f}), minimal allocation",
            )
        
        # Scale with uncertainty
        scale = (self.config.confidence_threshold - confidence) / self.config.confidence_threshold
        samples = int(self.config.min_samples + scale * (self.config.max_samples - self.config.min_samples))
        tokens = int(self.config.token_budget_base + scale * (self.config.token_budget_max - self.config.token_budget_base))
        
        reasoning = f"Low confidence ({confidence:.2f}), scaled allocation by {scale:.2f}"
        return (samples, tokens, reasoning)
    
    def _adaptive_allocation(
        self,
        uncertainty: float,
        confidence: float,
        step_type: str,
        step_context: dict,
    ) -> tuple[int, int, str]:
        """Adaptive allocation considering step type and history."""
        # Start with uncertainty-based allocation
        samples, tokens, base_reasoning = self._uncertainty_based_allocation(uncertainty, confidence)
        
        # Apply step type multiplier
        multiplier = self.config.step_multipliers.get(step_type, 1.0)
        tokens = int(tokens * multiplier)
        
        # Consider historical context if available
        if self._budget_history:
            recent_uncertainty = self._budget_history[-1]["uncertainty"] if self._budget_history else 0
            
            # If uncertainty is increasing, allocate more
            if uncertainty > recent_uncertainty * 1.2:
                tokens = int(tokens * 1.3)
                base_reasoning += ", increasing due to rising uncertainty"
        
        reasoning = f"{base_reasoning}, step='{step_type}' (×{multiplier})"
        return (samples, tokens, reasoning)
    
    def _default_uncertainty(self, votes: list) -> float:
        """Calculate uncertainty from vote distribution."""
        return self._uncertainty_quantifier.from_votes(votes)
    
    def _calculate_savings(self) -> dict:
        """Calculate token savings vs uniform allocation."""
        if not self._budget_history:
            return {"savings": 0, "percent": 0}
        
        uniform_tokens = len(self._budget_history) * self.config.token_budget_base
        actual_tokens = sum(h["tokens"] for h in self._budget_history)
        
        savings = uniform_tokens - actual_tokens
        percent = (savings / uniform_tokens * 100) if uniform_tokens > 0 else 0
        
        return {
            "savings": savings,
            "percent": round(percent, 1),
        }


class BudgetManager:
    """Manages compute budget across multiple research tasks.
    
    Provides higher-level budget management for long-running
    research sessions with multiple queries.
    """
    
    def __init__(
        self,
        total_budget: int = 50000,
        allocator_config: AllocationConfig | None = None,
    ):
        self.total_budget = total_budget
        self.allocator = CATTSAllocator(config=allocator_config)
        self._spent = 0
        self._task_history: list[dict] = []
    
    def check_budget(self, estimated_cost: int = 1000) -> bool:
        """Check if there's enough budget remaining."""
        return (self._spent + estimated_cost) <= self.total_budget
    
    def allocate_for_task(
        self,
        task_description: str,
        vote_distribution: list[Any] | None = None,
    ) -> AllocationDecision | None:
        """Allocate budget for a task, respecting global limits."""
        if not self.check_budget():
            return None
        
        if vote_distribution:
            decision = self.allocator.allocate(vote_distribution, step_type="search")
        else:
            # Initial allocation without votes
            initial = self.allocator.allocate_for_generation(task_description)
            decision = AllocationDecision(
                samples=initial["samples"],
                token_budget=initial["token_budget"],
                uncertainty=0.5,  # Unknown initially
                confidence=0.5,
                should_continue=True,
                reasoning="Initial allocation based on complexity estimate",
            )
        
        # Track spending
        self._spent += decision.token_budget
        self._task_history.append({
            "task": task_description[:50],
            "tokens": decision.token_budget,
            "confidence": decision.confidence,
        })
        
        return decision
    
    def get_status(self) -> dict:
        """Get budget status."""
        return {
            "total": self.total_budget,
            "spent": self._spent,
            "remaining": self.total_budget - self._spent,
            "percent_used": round(self._spent / self.total_budget * 100, 1),
            "tasks": len(self._task_history),
        }
