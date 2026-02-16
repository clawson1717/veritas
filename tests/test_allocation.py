"""Tests for CATTS allocator module."""

import pytest
from veritas.allocation.catts import CATTSAllocator, AllocationDecision


class TestCATTSAllocator:
    """Test suite for CATTS allocator."""
    
    def test_allocator_initialization(self):
        """Test allocator initialization."""
        allocator = CATTSAllocator()
        assert allocator.base_allocation == 100
        assert allocator.max_allocation == 1000
        assert allocator.min_allocation == 20
    
    def test_allocator_custom_config(self):
        """Test allocator with custom configuration."""
        allocator = CATTSAllocator(
            base_allocation=200,
            max_allocation=2000,
            min_allocation=50,
        )
        assert allocator.base_allocation == 200
        assert allocator.max_allocation == 2000
        assert allocator.min_allocation == 50
