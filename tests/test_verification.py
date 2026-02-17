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
        """Test verifier with default items."""
        verifier = ChecklistVerifier()
        assert len(verifier.items) == 0  # Default has no items
    
    def test_verifier_with_name(self):
        """Test verifier with custom name."""
        verifier = ChecklistVerifier(name="test")
        assert verifier.name == "test"
    
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


class TestChecklistItem:
    """Test suite for ChecklistItem."""
    
    def test_item_initialization(self):
        """Test item initialization."""
        item = ChecklistItem(
            id="test",
            description="Test check",
            check_fn=lambda x: True,
        )
        assert item.id == "test"
        assert item.description == "Test check"
        assert item.status == CheckStatus.PENDING
        assert item.required is True
        assert item.weight == 1.0
    
    def test_item_custom_values(self):
        """Test item with custom values."""
        item = ChecklistItem(
            id="custom",
            description="Custom check",
            check_fn=lambda x: x > 5,
            required=False,
            weight=2.0,
            category="test",
        )
        assert item.id == "custom"
        assert item.required is False
        assert item.weight == 2.0
        assert item.category == "test"
    
    def test_item_evaluate_pass(self):
        """Test item evaluation passes."""
        item = ChecklistItem(
            id="pass",
            description="Passing check",
            check_fn=lambda x: x > 5,
        )
        result = item.evaluate(10)
        assert result is True
        assert item.status == CheckStatus.PASSED
    
    def test_item_evaluate_fail(self):
        """Test item evaluation fails."""
        item = ChecklistItem(
            id="fail",
            description="Failing check",
            check_fn=lambda x: x > 5,
        )
        result = item.evaluate(3)
        assert result is False
        assert item.status == CheckStatus.FAILED
    
    def test_item_reset(self):
        """Test item reset."""
        item = ChecklistItem(
            id="test",
            description="Test",
            check_fn=lambda x: True,
        )
        item.evaluate(10)
        assert item.status == CheckStatus.PASSED
        item.reset()
        assert item.status == CheckStatus.PENDING


class TestChecklistResult:
    """Test suite for ChecklistResult."""
    
    def test_result_to_dict(self):
        """Test result conversion to dict."""
        from veritas.verification.checklist import ChecklistResult
        
        result = ChecklistResult(
            total=5,
            passed=4,
            failed=1,
            skipped=0,
            required_failed=0,
            all_passed=False,
            score=0.8,
        )
        d = result.to_dict()
        assert d["total"] == 5
        assert d["passed"] == 4
        assert d["score"] == 0.8
