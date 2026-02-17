"""Tests for CM2 verification module."""

import pytest
from veritas.verification.checklist import (
    ChecklistVerifier,
    ChecklistItem,
    CheckStatus,
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
        new_item = ChecklistItem(
            id="custom",
            description="Custom check",
            check_fn=lambda x: True,
            weight=1.5,
        )
        verifier.add_item(new_item)
        assert len(verifier.items) == 1
