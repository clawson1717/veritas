"""Tests for the AgentIntegrator."""

import pytest
from veritas.core.integrator import (
    AgentIntegrator,
    IntegrationConfig,
    IntegrationResult,
    create_integrator,
    create_minimal_integrator,
    create_full_integrator,
)


class TestIntegrationConfig:
    """Tests for IntegrationConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = IntegrationConfig()
        assert config.enable_routing == True
        assert config.enable_allocation == True
        assert config.enable_verification == True
        assert config.enable_refinement == True
        assert config.confidence_threshold == 0.8
        assert config.max_refinement_iterations == 5
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = IntegrationConfig(
            enable_routing=False,
            enable_verification=True,
            confidence_threshold=0.9,
        )
        assert config.enable_routing == False
        assert config.confidence_threshold == 0.9


class TestAgentIntegrator:
    """Tests for AgentIntegrator."""
    
    def test_create_integrator(self):
        """Test integrator creation."""
        integrator = AgentIntegrator()
        assert integrator is not None
        assert integrator.config is not None
    
    def test_create_with_config(self):
        """Test integrator with custom config."""
        config = IntegrationConfig(enable_routing=False)
        integrator = AgentIntegrator(config=config)
        assert integrator.config.enable_routing == False
    
    def test_reset(self):
        """Test integrator reset."""
        integrator = AgentIntegrator()
        integrator._current_query = "test query"
        integrator.reset()
        assert integrator._current_query == ""
    
    def test_get_history_empty(self):
        """Test getting empty history."""
        integrator = AgentIntegrator()
        history = integrator.get_history()
        assert history == []


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_integrator_defaults(self):
        """Test create_integrator with defaults."""
        integrator = create_integrator()
        assert integrator.config.enable_routing == True
        assert integrator.config.enable_verification == True
    
    def test_create_minimal_integrator(self):
        """Test minimal integrator creation."""
        integrator = create_minimal_integrator()
        assert integrator.config.enable_routing == False
        assert integrator.config.enable_allocation == False
        assert integrator.config.enable_verification == True
        assert integrator.config.enable_refinement == False
    
    def test_create_full_integrator(self):
        """Test full integrator creation."""
        integrator = create_full_integrator()
        assert integrator.config.enable_routing == True
        assert integrator.config.enable_allocation == True
        assert integrator.config.enable_verification == True
        assert integrator.config.enable_refinement == True
    
    def test_create_integrator_custom(self):
        """Test create_integrator with custom params."""
        integrator = create_integrator(
            enable_routing=False,
            confidence_threshold=0.95,
        )
        assert integrator.config.enable_routing == False
        assert integrator.config.confidence_threshold == 0.95


class TestIntegrationResult:
    """Tests for IntegrationResult."""
    
    def test_successful_result(self):
        """Test successful integration result."""
        result = IntegrationResult(
            success=True,
            answer="Test answer",
            confidence=0.9,
        )
        assert result.success == True
        assert result.answer == "Test answer"
        assert result.confidence == 0.9
        assert result.error is None
    
    def test_failed_result(self):
        """Test failed integration result."""
        result = IntegrationResult(
            success=False,
            answer="",
            error="Test error",
        )
        assert result.success == False
        assert result.error == "Test error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
