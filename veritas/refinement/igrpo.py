"""iGRPO: Iterative GRPO with Self-Feedback Refinement.

Implements dynamic self-conditioning where the model learns to refine
its own best attempts based on uncertainty signals from previous iterations.

Key components:
- SelfFeedbackRefiner: Main refinement loop that iteratively improves responses
- ConvergenceChecker: Determines when refinement is no longer improving
- RefinementResult: Structured output containing original, refined, and metadata

The approach draws from the iGRPO paper's insight that training models
to refine their own outputs (rather than just generating once) can
improve quality without additional external feedback.
"""

import os
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from openai import AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RefinementResult:
    """Result of a self-feedback refinement process.
    
    Attributes:
        original: The original response before refinement
        refined: The refined response after iterative improvement
        iterations: Number of refinement iterations performed
        converged: Whether the refinement converged (stopped improving)
        improvement_score: Score indicating how much the refinement improved
        confidence_before: Confidence score before refinement
        confidence_after: Confidence score after refinement
        refinement_history: List of intermediate refinements
    """
    original: str
    refined: str
    iterations: int
    converged: bool
    improvement_score: float
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    refinement_history: list[str] = field(default_factory=list)
    
    @property
    def best_response(self) -> str:
        """Return the best version (original or refined)."""
        if self.improvement_score > 0:
            return self.refined
        return self.original
    
    @property
    def was_improved(self) -> bool:
        """Whether refinement improved the response."""
        return self.improvement_score > 0


# =============================================================================
# Similarity / Improvement Metrics
# =============================================================================

class SimilarityMetric:
    """Computes similarity between text responses for improvement detection."""
    
    def __init__(self, method: str = "tfidf"):
        """Initialize similarity metric.
        
        Args:
            method: Similarity method ("tfidf" or "jaccard")
        """
        self.method = method
        self._vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
        )
    
    def compute(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score in [0, 1], where 1 = identical
        """
        if not text1 or not text2:
            return 0.0
        
        if self.method == "jaccard":
            return self._jaccard_similarity(text1, text2)
        return self._tfidf_similarity(text1, text2)
    
    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """Compute TF-IDF cosine similarity."""
        try:
            vectors = self._vectorizer.fit_transform([text1, text2])
            sim_matrix = cosine_similarity(vectors)
            return float(sim_matrix[0, 1])
        except Exception:
            return self._jaccard_similarity(text1, text2)
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity of word sets."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def improvement(
        self,
        original: str,
        refined: str,
        reference: str | None = None,
    ) -> float:
        """Calculate improvement score between original and refined.
        
        Args:
            original: Original text
            refined: Refined text
            reference: Optional reference for comparison
            
        Returns:
            Improvement score (-1 to 1). Positive means improvement.
        """
        # Base similarity
        base_sim = self.compute(original, refined)
        
        # If texts are very similar, no significant change
        if base_sim > 0.95:
            return 0.0
        
        # If we have a reference, measure improvement relative to it
        if reference:
            refined_vs_ref = self.compute(refined, reference)
            orig_vs_ref = self.compute(original, reference)
            return refined_vs_ref - orig_vs_ref
        
        # Length change as a simple heuristic
        # Some improvements involve better structuring/expanding
        len_ratio = len(refined) / max(len(original), 1)
        
        # Penalize if significantly shorter, allow moderate expansion
        length_factor = min(len_ratio, 1.5) / 1.5
        
        return (base_sim * length_factor) - 0.5


# =============================================================================
# Convergence Checker
# =============================================================================

@dataclass
class ConvergenceConfig:
    """Configuration for convergence checking."""
    max_iterations: int = 5
    min_improvement_threshold: float = 0.01
    similarity_threshold: float = 0.98
    patience: int = 2  # Stop if no improvement for N consecutive iterations


class ConvergenceChecker:
    """Determines when refinement has converged (stopped improving).
    
    Uses multiple signals:
    - Similarity between consecutive refinements (stable = converged)
    - Improvement score below threshold
    - Patience counter for early stopping
    
    Example:
        >>> checker = ConvergenceChecker()
        >>> checker.update(similarity=0.99, improvement=0.001)
        >>> print(checker.has_converged())  # True
    """
    
    def __init__(self, config: ConvergenceConfig | None = None):
        """Initialize convergence checker.
        
        Args:
            config: Configuration for convergence behavior
        """
        self.config = config or ConvergenceConfig()
        self._iteration = 0
        self._similarities: list[float] = []
        self._improvements: list[float] = []
        self._no_improvement_count = 0
    
    def update(self, similarity: float, improvement: float) -> None:
        """Update convergence state with new iteration results.
        
        Args:
            similarity: Similarity between current and previous version
            improvement: Improvement score of current iteration
        """
        self._iteration += 1
        self._similarities.append(similarity)
        self._improvements.append(improvement)
        
        # Track consecutive non-improving iterations
        if improvement <= self.config.min_improvement_threshold:
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0
    
    def has_converged(self) -> bool:
        """Check if refinement has converged.
        
        Returns:
            True if refinement should stop
        """
        cfg = self.config
        
        # Max iterations reached
        if self._iteration >= cfg.max_iterations:
            return True
        
        # Not enough iterations to judge convergence
        if self._iteration < 2:
            return False
        
        # Check similarity threshold (stable = converged)
        if self._similarities[-1] >= cfg.similarity_threshold:
            return True
        
        # Check patience (no improvement for too long)
        if self._no_improvement_count >= cfg.patience:
            return True
        
        return False
    
    def reset(self) -> None:
        """Reset convergence state."""
        self._iteration = 0
        self._similarities.clear()
        self._improvements.clear()
        self._no_improvement_count = 0
    
    @property
    def iteration(self) -> int:
        """Current iteration count."""
        return self._iteration
    
    @property
    def last_improvement(self) -> float:
        """Most recent improvement score."""
        return self._improvements[-1] if self._improvements else 0.0


# =============================================================================
# Self-Feedback Refiner
# =============================================================================

class SelfFeedbackRefiner:
    """Implements iterative self-feedback refinement (iGRPO).
    
    Takes an initial response and iteratively refines it using:
    - The model's own reasoning about potential improvements
    - Uncertainty signals to decide when further refinement is needed
    - Convergence checking to avoid infinite loops
    
    Key insight from iGRPO: Train models to generate refinement critiques
    of their own outputs, then apply those critiques for improvement.
    
    Example:
        >>> refiner = SelfFeedbackRefiner(model="gpt-4o-mini")
        >>> result = await refiner.refine(
        ...     initial_response="The answer is 42.",
        ...     context={"question": "What is the meaning of life?"}
        ... )
        >>> print(result.best_response)
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        convergence_config: ConvergenceConfig | None = None,
    ):
        """Initialize the self-feedback refiner.
        
        Args:
            model: LLM model to use for refinement generation
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            max_tokens: Maximum tokens for refinement output
            temperature: Temperature for generation
            convergence_config: Configuration for convergence checking
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        self._similarity = SimilarityMetric()
        self._convergence = ConvergenceConfig() if convergence_config is None else convergence_config
        self._refine_prompt_template = self._default_refine_prompt
    
    def _default_refine_prompt(self, response: str, context: dict[str, Any]) -> str:
        """Default prompt for generating refinement suggestions.
        
        Args:
            response: The current response to refine
            context: Additional context about the original task
            
        Returns:
            Formatted prompt for the LLM
        """
        question = context.get("question", "the task")
        constraints = context.get("constraints", "")
        
        return f"""You are a self-critique assistant. Your task is to refine your own previous response to improve its quality.

Original Question: {question}

Current Response:
---
{response}
---

{constraints}

Instructions:
1. Identify 2-3 specific issues or areas for improvement in the current response
2. Provide a refined version that addresses these issues
3. Keep the refined version accurate - don't hallucinate new information

If the response is already good, you may return it unchanged with minor edits.

Refined Response:"""
    
    async def refine(
        self,
        initial_response: str,
        context: dict[str, Any],
        refinement_prompt_fn: Callable[[str, dict[str, Any]], str] | None = None,
    ) -> RefinementResult:
        """Refine an initial response using iterative self-feedback.
        
        The refinement process:
        1. Generate potential improvements using the LLM
        2. Apply improvements to create a refined version
        3. Check for convergence (no more improvement possible)
        4. Repeat until convergence or max iterations
        
        Args:
            initial_response: The original response to refine
            context: Context about the original task (question, constraints, etc.)
            refinement_prompt_fn: Custom prompt function (optional)
            
        Returns:
            RefinementResult with original, refined, and metadata
        """
        if not initial_response or not initial_response.strip():
            return RefinementResult(
                original=initial_response,
                refined=initial_response,
                iterations=0,
                converged=True,
                improvement_score=0.0,
            )
        
        # Get prompt function
        prompt_fn = refinement_prompt_fn or self._default_refine_prompt
        
        # Initialize tracking
        current = initial_response
        history = [initial_response]
        checker = ConvergenceChecker(self._convergence)
        
        # Initial confidence estimation
        confidence_before = await self._estimate_confidence(current, context)
        
        # Iterative refinement
        for i in range(self._convergence.max_iterations):
            # Generate refinement
            refined = await self._generate_refinement(current, context, prompt_fn)
            
            if not refined or refined.strip() == current.strip():
                # No meaningful refinement generated
                checker.update(similarity=1.0, improvement=0.0)
            else:
                # Calculate metrics
                similarity = self._similarity.compute(current, refined)
                improvement = self._similarity.improvement(current, refined)
                
                checker.update(similarity=similarity, improvement=improvement)
                
                # Update current if improved
                if improvement > self._convergence.min_improvement_threshold:
                    current = refined
                    history.append(refined)
            
            # Check convergence
            if checker.has_converged():
                break
        
        # Final confidence
        confidence_after = await self._estimate_confidence(current, context)
        
        # Calculate final improvement
        final_improvement = self._similarity.improvement(
            initial_response, 
            current,
            reference=context.get("reference"),
        )
        
        return RefinementResult(
            original=initial_response,
            refined=current,
            iterations=checker.iteration,
            converged=checker.has_converged(),
            improvement_score=final_improvement,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            refinement_history=history,
        )
    
    async def _generate_refinement(
        self,
        current_response: str,
        context: dict[str, Any],
        prompt_fn: Callable[[str, dict[str, Any]], str],
    ) -> str:
        """Generate a refined version of the current response.
        
        Args:
            current_response: Current version to refine
            context: Task context
            prompt_fn: Prompt generation function
            
        Returns:
            Refined response text
        """
        prompt = prompt_fn(current_response, context)
        
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a meticulous editor focused on improving response quality. "
                                   "Provide specific, actionable improvements."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            content = response.choices[0].message.content
            return content.strip() if content else current_response
            
        except Exception as e:
            # On error, return current response unchanged
            return current_response
    
    async def _estimate_confidence(
        self,
        response: str,
        context: dict[str, Any],
    ) -> float:
        """Estimate confidence in a response.
        
        Uses heuristic signals:
        - Response length (too short = low confidence)
        - Presence of hedge words (might indicate uncertainty)
        - Presence of factual markers
        
        Args:
            response: The response to evaluate
            context: Task context
            
        Returns:
            Confidence score in [0, 1]
        """
        if not response:
            return 0.0
        
        score = 0.5  # Base confidence
        
        # Length factor
        length = len(response)
        if length > 100:
            score += 0.1
        if length > 500:
            score += 0.1
        
        # Hedge words (lower confidence)
        hedges = ["might", "could", "possibly", "perhaps", "probably", 
                  "might be", "could be", "unclear", "uncertain"]
        hedge_count = sum(1 for h in hedges if h in response.lower())
        score -= hedge_count * 0.05
        
        # Factual markers (higher confidence)
        factual = ["according to", "research shows", "data indicates",
                   "studies show", "evidence suggests"]
        factual_count = sum(1 for f in factual if f in response.lower())
        score += factual_count * 0.1
        
        return float(np.clip(score, 0.0, 1.0))
    
    def should_refine(
        self,
        confidence: float,
        uncertainty: float | None = None,
    ) -> bool:
        """Decide whether to refine based on confidence/uncertainty signals.
        
        This integrates with the CATTS uncertainty from earlier steps.
        
        Args:
            confidence: Current confidence score (from verification/uncertainty)
            uncertainty: Optional explicit uncertainty score
            
        Returns:
            True if refinement should be attempted
        """
        # Use uncertainty if provided, otherwise use 1 - confidence
        effective_uncertainty = uncertainty if uncertainty is not None else (1.0 - confidence)
        
        # Refine if uncertainty is above threshold
        return effective_uncertainty > 0.3


# =============================================================================
# Convenience Functions
# =============================================================================

async def refine_response(
    response: str,
    question: str,
    model: str = "gpt-4o-mini",
    **kwargs,
) -> RefinementResult:
    """Convenience function to refine a response.
    
    Args:
        response: The response to refine
        question: The original question/topic
        model: Model to use for refinement
        **kwargs: Additional arguments for SelfFeedbackRefiner
        
    Returns:
        RefinementResult
    """
    refiner = SelfFeedbackRefiner(model=model, **kwargs)
    return await refiner.refine(
        initial_response=response,
        context={"question": question},
    )


# Export public API
__all__ = [
    "RefinementResult",
    "SelfFeedbackRefiner",
    "ConvergenceChecker",
    "ConvergenceConfig",
    "SimilarityMetric",
    "refine_response",
]
