"""Tests for CATTS allocator module."""

import pytest
from veritas.allocation.catts import CATTSAllocator, AllocationConfig, AllocationStrategy


class TestCATTSAllocator:
    """Test suite for CATTS allocator."""
    
    def test_allocator_initialization(self):
        """Test allocator initialization."""
        allocator = CATTSAllocator()
        assert allocator.config is not None
        assert allocator.config.min_samples == 3
        assert allocator.config.max_samples == 10
    
    def test_allocator_with_config(self):
        """Test allocator with custom configuration."""
        config = AllocationConfig(
            min_samples=5,
            max_samples=15,
            token_budget_base=2000,
            token_budget_max=8000,
        )
        allocator = CATTSAllocator(config=config)
        assert allocator.config.min_samples == 5
        assert allocator.config.max_samples == 15
        assert allocator.config.token_budget_base == 2000
        assert allocator.config.token_budget_max == 8000
    
    def test_allocator_with_strategy(self):
        """Test allocator with custom strategy."""
        config = AllocationConfig(strategy=AllocationStrategy.UNIFORM)
        allocator = CATTSAllocator(config=config)
        assert allocator.config.strategy == AllocationStrategy.UNIFORM


class TestAllocationConfig:
    """Test suite for AllocationConfig."""
    
    def test_config_defaults(self):
        """Test config default values."""
        config = AllocationConfig()
        assert config.min_samples == 3
        assert config.max_samples == 10
        assert config.confidence_threshold == 0.8
        assert config.token_budget_base == 1000
        assert config.token_budget_max == 5000
        assert config.strategy == AllocationStrategy.ADAPTIVE
    
    def test_config_custom_values(self):
        """Test config with custom values."""
        config = AllocationConfig(
            min_samples=1,
            max_samples=20,
            confidence_threshold=0.9,
            token_budget_base=500,
            token_budget_max=10000,
            strategy=AllocationStrategy.UNCERTAINTY_BASED,
        )
        assert config.min_samples == 1
        assert config.max_samples == 20
        assert config.confidence_threshold == 0.9
        assert config.token_budget_base == 500
        assert config.token_budget_max == 10000
        assert config.strategy == AllocationStrategy.UNCERTAINTY_BASED
