"""Tests for uncertainty quantification module."""

import sys
sys.path.insert(0, '/home/corbin/.openclaw/workspace/projects/veritas')

import numpy as np

# Mock AsyncOpenAI for testing
class MockAsyncOpenAI:
    pass

# Patch before importing
import veritas.core.uncertainty as uncertainty_module
uncertainty_module.AsyncOpenAI = MockAsyncOpenAI

from veritas.core.uncertainty import (
    UncertaintyQuantifier,
    normalize_uncertainty,
    uncertainty_to_confidence,
)


def test_entropy_calculation():
    """Test entropy-based uncertainty."""
    uq = UncertaintyQuantifier(use_semantic_embeddings=False)
    
    # Uniform distribution = max uncertainty
    uniform = [0.25, 0.25, 0.25, 0.25]
    assert uq.from_entropy(uniform) == 1.0, "Uniform distribution should have max entropy"
    
    # Deterministic = no uncertainty
    deterministic = [1.0]
    assert uq.from_entropy(deterministic) == 0.0, "Deterministic should have no entropy"
    
    # Skewed distribution
    skewed = [0.8, 0.1, 0.1]
    entropy = uq.from_entropy(skewed)
    assert 0 < entropy < 1, f"Skewed entropy should be in (0, 1), got {entropy}"
    
    print("✓ Entropy calculation tests passed")


def test_variance_calculation():
    """Test variance-based uncertainty."""
    uq = UncertaintyQuantifier(use_semantic_embeddings=False)
    
    # All same = no uncertainty
    same = [5.0, 5.0, 5.0]
    assert uq.from_variance(same) == 0.0, "Identical values should have 0 variance"
    
    # Empty list
    assert uq.from_variance([]) == 1.0, "Empty list should have max uncertainty"
    
    # Single value
    assert uq.from_variance([42.0]) == 0.0, "Single value should have 0 variance"
    
    # Varying values
    varying = [1.0, 2.0, 3.0, 4.0, 5.0]
    var_unc = uq.from_variance(varying)
    assert 0 < var_unc < 1, f"Variance uncertainty should be in (0, 1), got {var_unc}"
    
    print("✓ Variance calculation tests passed")


def test_categorical_votes():
    """Test uncertainty from categorical votes."""
    uq = UncertaintyQuantifier(use_semantic_embeddings=False)
    
    # All same
    all_yes = ["yes", "yes", "yes"]
    assert uq.from_votes(all_yes) == 0.0, "Unanimous votes should have 0 uncertainty"
    
    # Empty
    assert uq.from_votes([]) == 1.0, "Empty votes should have max uncertainty"
    
    # Mixed votes
    mixed = ["yes", "no", "yes", "no"]
    unc = uq.from_votes(mixed)
    assert unc > 0, f"Mixed votes should have uncertainty > 0, got {unc}"
    
    # Booleans
    bool_votes = [True, True, False]
    bool_unc = uq.from_votes(bool_votes)
    assert 0 < bool_unc <= 1, f"Boolean votes uncertainty should be in (0, 1], got {bool_unc}"
    
    print("✓ Categorical votes tests passed")


def test_numerical_votes():
    """Test uncertainty from numerical votes."""
    uq = UncertaintyQuantifier(use_semantic_embeddings=False)
    
    # All same
    same = [42.0, 42.0, 42.0]
    assert uq.from_votes(same) == 0.0, "Identical numeric votes should have 0 uncertainty"
    
    # Varying
    varying = [10.0, 20.0, 30.0]
    unc = uq.from_votes(varying)
    assert 0 < unc < 1, f"Varying numeric votes should have uncertainty in (0, 1), got {unc}"
    
    print("✓ Numerical votes tests passed")


def test_semantic_agreement():
    """Test semantic disagreement calculation."""
    uq = UncertaintyQuantifier(use_semantic_embeddings=False)
    
    # Empty
    assert uq.from_agreement([]) == 1.0, "Empty texts should have max disagreement"
    
    # Single text
    assert uq.from_agreement(["hello"]) == 0.0, "Single text should have 0 disagreement"
    
    # Similar texts (should have low disagreement)
    similar = [
        "The sky is blue today",
        "Today the sky is blue",
        "Blue sky today"
    ]
    similar_unc = uq.from_agreement(similar)
    assert 0 <= similar_unc < 0.8, f"Similar texts should have low disagreement, got {similar_unc}"
    
    # Different texts (should have higher disagreement)
    different = [
        "Quantum mechanics is fascinating",
        "I love pizza with pepperoni",
        "The stock market crashed yesterday"
    ]
    diff_unc = uq.from_agreement(different)
    assert diff_unc > similar_unc, f"Different texts should have higher disagreement than similar"
    
    print("✓ Semantic agreement tests passed")


def test_helper_functions():
    """Test helper utility functions."""
    # Normalize uncertainty
    assert normalize_uncertainty(0.5, "entropy") == 0.5
    assert normalize_uncertainty(-0.1, "entropy") == 0.0
    assert normalize_uncertainty(1.5, "entropy") == 1.0
    
    # Confidence conversion
    assert uncertainty_to_confidence(0.0) == 1.0
    assert uncertainty_to_confidence(1.0) == 0.0
    assert uncertainty_to_confidence(0.5) == 0.5
    assert uncertainty_to_confidence(-0.1) == 1.0  # Clipped
    assert uncertainty_to_confidence(1.5) == 0.0   # Clipped
    
    print("✓ Helper functions tests passed")


def test_confidence_from_votes():
    """Test confidence calculation from votes."""
    uq = UncertaintyQuantifier(use_semantic_embeddings=False)
    
    # All same = high confidence
    assert uq.confidence_from_votes(["yes", "yes"]) == 1.0
    
    # Empty = zero confidence
    assert uq.confidence_from_votes([]) == 0.0
    
    # Mixed = some confidence
    mixed_conf = uq.confidence_from_votes(["yes", "no"])
    assert 0 <= mixed_conf < 1, f"Mixed votes should have confidence < 1, got {mixed_conf}"
    
    print("✓ Confidence from votes tests passed")


if __name__ == "__main__":
    print("\nRunning uncertainty quantification tests...\n")
    
    test_entropy_calculation()
    test_variance_calculation()
    test_categorical_votes()
    test_numerical_votes()
    test_semantic_agreement()
    test_helper_functions()
    test_confidence_from_votes()
    
    print("\n✅ All tests passed!")
