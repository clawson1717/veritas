"""Tests for CM2 verification module."""

import pytest
from veritas.verification.checklist import (
    ChecklistVerifier,
    Criterion,
    CriterionStatus,
)


class TestChecklistVerifier:
    """Test suite for ChecklistVerifier."""
    
    def test_verifier_initialization(self):
        """Test verifier with default criteria."""
        verifier = ChecklistVerifier()
        assert len(verifier.criteria) == 6  # Default criteria
        assert verifier.min_pass_threshold == 0.8
    
    def test_verifier_custom_threshold(self):
        """Test verifier with custom threshold."""
        verifier = ChecklistVerifier(min_pass_threshold=0.9)
        assert verifier.min_pass_threshold == 0.9
    
    def test_add_custom_criterion(self):
        """Test adding custom criterion."""
        verifier = ChecklistVerifier()
        new_criterion = Criterion(
            id="custom",
            description="Custom check",
            weight=1.5,
        )
        verifier.add_criterion(new_criterion)
        assert len(verifier.criteria) == 7
