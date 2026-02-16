"""Uncertainty quantification module for Veritas.

Provides tools for quantifying uncertainty from vote distributions,
supporting the CATTS (Compute Allocation via Test-Time Scaling) approach.
Uses multiple metrics including entropy, variance, and semantic disagreement.
"""

import os
import hashlib
from typing import Any
from collections import Counter

import numpy as np
from openai import AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_uncertainty(raw: float, method: str = "entropy") -> float:
    """Normalize raw uncertainty value to [0, 1] range.
    
    Args:
        raw: Raw uncertainty value
        method: Normalization method to use ("entropy", "variance", "agreement")
        
    Returns:
        Normalized uncertainty in [0, 1]
    """
    if method == "entropy":
        # Entropy is naturally in [0, 1] when normalized by max entropy
        return float(np.clip(raw, 0.0, 1.0))
    elif method == "variance":
        # Coefficient of variation can exceed 1, so we clip
        return float(np.clip(raw, 0.0, 1.0))
    elif method == "agreement":
        # Agreement is typically cosine similarity-based
        # Invert so 1 = max disagreement (uncertainty)
        return float(np.clip(1.0 - raw, 0.0, 1.0))
    else:
        # Default: simple clipping
        return float(np.clip(raw, 0.0, 1.0))


def uncertainty_to_confidence(uncertainty: float) -> float:
    """Convert uncertainty to CATTS-compatible confidence score.
    
    Args:
        uncertainty: Uncertainty value in [0, 1]
        
    Returns:
        Confidence score in [0, 1] where 1 = high confidence
    """
    return float(np.clip(1.0 - uncertainty, 0.0, 1.0))


class UncertaintyQuantifier:
    """Quantifies uncertainty from various sources.
    
    Implements multiple uncertainty metrics:
    - Vote entropy: Shannon entropy for categorical votes
    - Normalized variance: Coefficient of variation for numerical values
    - Semantic disagreement: Embedding-based text disagreement
    
    All methods return uncertainty in [0, 1] where:
    - 0.0 = completely certain (all samples agree)
    - 1.0 = maximally uncertain (maximal disagreement)
    
    Example:
        >>> uq = UncertaintyQuantifier()
        >>> uncertainty = uq.from_votes(["yes", "yes", "no"])
        >>> confidence = 1.0 - uncertainty
    """
    
    def __init__(self, use_semantic_embeddings: bool = True):
        """Initialize the uncertainty quantifier.
        
        Args:
            use_semantic_embeddings: Whether to use sentence-transformers
                                    for semantic disagreement (if available)
        """
        self.use_semantic_embeddings = use_semantic_embeddings
        self._embedding_model = None
        
        # Try to load sentence-transformers if requested
        if use_semantic_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                # Use a lightweight model for efficiency
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                # Fall back to TF-IDF vectors
                self.use_semantic_embeddings = False
    
    def from_votes(self, votes: list[Any]) -> float:
        """Calculate uncertainty from multiple vote samples.
        
        Uses normalized Shannon entropy for categorical votes.
        For text votes, uses semantic disagreement.
        For numerical votes, uses coefficient of variation.
        
        Args:
            votes: List of vote samples (can be strings, numbers, or booleans)
            
        Returns:
            Uncertainty score in [0, 1]
        """
        if not votes:
            return 1.0  # Max uncertainty for empty votes
        
        if len(votes) == 1:
            return 0.0  # Certain with only one sample
        
        # Detect vote type
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in votes):
            # Numerical votes
            return self.from_variance([float(v) for v in votes])
        elif all(isinstance(v, str) for v in votes):
            # Text votes - use semantic disagreement
            return self.from_agreement(votes)
        else:
            # Categorical votes (including booleans)
            return self._categorical_entropy(votes)
    
    def _categorical_entropy(self, votes: list[Any]) -> float:
        """Calculate normalized Shannon entropy for categorical votes.
        
        Args:
            votes: List of categorical vote samples
            
        Returns:
            Normalized entropy in [0, 1]
        """
        if not votes:
            return 1.0
        
        # Count occurrences
        counter = Counter(votes)
        total = len(votes)
        
        # Calculate probabilities
        probs = [count / total for count in counter.values()]
        
        return self.from_entropy(probs)
    
    def from_entropy(self, probs: list[float]) -> float:
        """Calculate normalized entropy-based uncertainty.
        
        Uses Shannon entropy normalized by the maximum possible entropy
        for the given number of categories.
        
        Args:
            probs: List of probability values (should sum to ~1)
            
        Returns:
            Normalized entropy in [0, 1] where 1 = uniform distribution
        """
        if not probs:
            return 1.0
        
        probs = np.array(probs)
        # Remove zero probabilities to avoid log(0)
        probs = probs[probs > 0]
        
        if len(probs) == 0:
            return 1.0
        
        # Normalize to ensure sum to 1
        probs = probs / probs.sum()
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        # Normalize by max entropy (log2 of number of categories)
        max_entropy = np.log2(len(probs))
        
        if max_entropy == 0:
            return 0.0
        
        normalized = entropy / max_entropy
        return float(np.clip(normalized, 0.0, 1.0))
    
    def from_variance(self, values: list[float]) -> float:
        """Calculate coefficient of variation as uncertainty.
        
        Uses the coefficient of variation (std/mean) normalized
        to [0, 1] range for uncertainty quantification.
        
        Args:
            values: List of numerical values
            
        Returns:
            Normalized variance-based uncertainty in [0, 1]
        """
        if not values:
            return 1.0
        
        values_arr = np.array(values, dtype=float)
        
        if len(values_arr) < 2:
            return 0.0
        
        mean = np.mean(values_arr)
        std = np.std(values_arr, ddof=1)  # Sample std
        
        if mean == 0:
            # If mean is 0, use std directly normalized
            return float(np.clip(std / (std + 1), 0.0, 1.0))
        
        # Coefficient of variation
        cv = std / abs(mean)
        
        # Normalize: CV can theoretically be > 1, so we use a soft clipping
        # cv_normalized = cv / (1 + cv) maps [0, inf) to [0, 1)
        cv_normalized = cv / (1 + cv)
        
        return float(np.clip(cv_normalized, 0.0, 1.0))
    
    def from_agreement(self, texts: list[str]) -> float:
        """Calculate semantic disagreement from text samples.
        
        Uses embeddings to measure how semantically different
the texts are. Returns high uncertainty when texts are diverse.
        
        Args:
            texts: List of text samples
            
        Returns:
            Disagreement score in [0, 1] where 1 = high disagreement
        """
        if not texts:
            return 1.0
        
        if len(texts) == 1:
            return 0.0
        
        # Clean and deduplicate (empty strings cause issues)
        texts = [t.strip() for t in texts if t.strip()]
        if not texts:
            return 1.0
        
        # Get embeddings
        if self.use_semantic_embeddings and self._embedding_model is not None:
            embeddings = self._embedding_model.encode(texts)
        else:
            # Fall back to TF-IDF vectors
            embeddings = self._get_tfidf_embeddings(texts)
        
        if len(embeddings) < 2:
            return 0.0
        
        # Calculate pairwise cosine similarities
        similarities = cosine_similarity(embeddings)
        
        # Get upper triangle (excluding diagonal)
        upper_tri_indices = np.triu_indices(len(embeddings), k=1)
        pairwise_sims = similarities[upper_tri_indices]
        
        if len(pairwise_sims) == 0:
            return 0.0
        
        # Average similarity (agreement)
        avg_similarity = np.mean(pairwise_sims)
        
        # Convert to disagreement (uncertainty)
        # Higher disagreement = lower average similarity
        disagreement = 1.0 - avg_similarity
        
        return float(np.clip(disagreement, 0.0, 1.0))
    
    def _get_tfidf_embeddings(self, texts: list[str]) -> np.ndarray:
        """Get TF-IDF embeddings as fallback.
        
        Args:
            texts: List of text strings
            
        Returns:
            TF-IDF vectors as numpy array
        """
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        
        try:
            vectors = vectorizer.fit_transform(texts)
            return vectors.toarray()
        except Exception:
            # If TF-IDF fails, return simple bag-of-words
            return self._simple_embeddings(texts)
    
    def _simple_embeddings(self, texts: list[str]) -> np.ndarray:
        """Create simple hash-based embeddings as ultimate fallback.
        
        Args:
            texts: List of text strings
            
        Returns:
            Simple feature vectors
        """
        # Create simple feature vectors based on character n-grams
        vectors = []
        for text in texts:
            # Create a simple feature vector
            features = []
            text_lower = text.lower()
            
            # Character bigram frequencies (simplified)
            for i in range(len(text_lower) - 1):
                bigram = text_lower[i:i+2]
                features.append(hash(bigram) % 100)
            
            # Pad or truncate to fixed length
            vec = np.zeros(100)
            for i, f in enumerate(features[:100]):
                vec[i] = f / 100.0  # Normalize
            
            vectors.append(vec)
        
        return np.array(vectors)
    
    def confidence_from_votes(self, votes: list[Any]) -> float:
        """Get confidence score from votes (convenience method).
        
        Args:
            votes: List of vote samples
            
        Returns:
            Confidence score in [0, 1]
        """
        uncertainty = self.from_votes(votes)
        return uncertainty_to_confidence(uncertainty)


class VoteSampler:
    """Generates multiple reasoning samples for uncertainty estimation.
    
    Implements the sampling component of CATTS by generating
    multiple LLM responses with higher temperature for diversity.
    
    Example:
        >>> sampler = VoteSampler(model="gpt-4o-mini")
        >>> votes = await sampler.sample("What causes rain?", n=5, temperature=0.7)
        >>> print(len(votes))  # 5
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_tokens: int = 500,
    ):
        """Initialize the vote sampler.
        
        Args:
            model: LLM model to use for sampling
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            max_tokens: Maximum tokens per sample
        """
        self.model = model
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    async def sample(
        self,
        prompt: str,
        n: int = 5,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> list[str]:
        """Generate multiple samples from the LLM.
        
        Uses higher temperature to encourage diversity in responses,
        which is essential for meaningful uncertainty estimation.
        
        Args:
            prompt: The prompt to send to the LLM
            n: Number of samples to generate
            temperature: Sampling temperature (higher = more diverse)
            system_prompt: Optional system prompt override
            
        Returns:
            List of n response strings
        """
        if n <= 0:
            return []
        
        system = system_prompt or (
            "You are a helpful research assistant. Answer the question "
            "concisely and directly."
        )
        
        # Generate samples concurrently
        tasks = [
            self._generate_single(prompt, temperature, system)
            for _ in range(n)
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        valid_responses = []
        for r in responses:
            if isinstance(r, Exception):
                # Log error but continue with other samples
                continue
            valid_responses.append(r)
        
        return valid_responses
    
    async def _generate_single(
        self,
        prompt: str,
        temperature: float,
        system_prompt: str,
    ) -> str:
        """Generate a single sample.
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature
            system_prompt: System prompt
            
        Returns:
            Generated response text
        """
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        
        content = response.choices[0].message.content
        return content.strip() if content else ""
    
    async def sample_with_confidence(
        self,
        prompt: str,
        n: int = 5,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> tuple[list[str], float]:
        """Generate samples and return them with aggregate confidence.
        
        Args:
            prompt: The prompt to send to the LLM
            n: Number of samples to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt override
            
        Returns:
            Tuple of (samples, confidence_score)
        """
        samples = await self.sample(prompt, n, temperature, system_prompt)
        
        if not samples:
            return [], 0.0
        
        # Calculate confidence using uncertainty quantifier
        uq = UncertaintyQuantifier()
        confidence = uq.confidence_from_votes(samples)
        
        return samples, confidence


# Import asyncio at the end to avoid circular import issues
import asyncio
