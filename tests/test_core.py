"""Tests for Veritas core modules."""

import pytest
from veritas.core.agent import ResearchAgent, ResearchQuery


class TestResearchAgent:
    """Test suite for ResearchAgent."""
    
    def test_agent_initialization(self):
        """Test agent can be initialized."""
        agent = ResearchAgent()
        assert agent.max_compute_budget == 1000
        assert agent.verification_threshold == 0.8
    
    def test_agent_custom_config(self):
        """Test agent with custom configuration."""
        agent = ResearchAgent(
            max_compute_budget=500,
            verification_threshold=0.9,
        )
        assert agent.max_compute_budget == 500
        assert agent.verification_threshold == 0.9


class TestResearchQuery:
    """Test suite for ResearchQuery."""
    
    def test_query_defaults(self):
        """Test query default values."""
        query = ResearchQuery(question="Test question")
        assert query.question == "Test question"
        assert query.depth == 3
        assert query.max_steps == 10
    
    def test_query_custom_values(self):
        """Test query with custom values."""
        query = ResearchQuery(
            question="Test",
            depth=5,
            max_steps=20,
        )
        assert query.depth == 5
        assert query.max_steps == 20
