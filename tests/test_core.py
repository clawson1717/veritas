"""Tests for Veritas core modules."""

import pytest
from veritas.core.agent import ResearchAgent, ResearchTask


class TestResearchAgent:
    """Test suite for ResearchAgent."""
    
    def test_agent_initialization(self):
        """Test agent can be initialized."""
        agent = ResearchAgent()
        assert agent.model == "gpt-4o-mini"
        assert agent.max_iterations == 3
        assert agent.enable_verification is True
    
    def test_agent_custom_config(self):
        """Test agent with custom configuration."""
        agent = ResearchAgent(
            model="gpt-4o",
            max_iterations=5,
            enable_verification=False,
        )
        assert agent.model == "gpt-4o"
        assert agent.max_iterations == 5
        assert agent.enable_verification is False


class TestResearchTask:
    """Test suite for ResearchTask."""
    
    def test_task_defaults(self):
        """Test task default values."""
        task = ResearchTask(query="Test question")
        assert task.query == "Test question"
        assert task.depth == "medium"
        assert task.max_steps == 10
    
    def test_task_custom_values(self):
        """Test task with custom values."""
        task = ResearchTask(
            query="Test",
            depth="deep",
            max_steps=20,
        )
        assert task.depth == "deep"
        assert task.max_steps == 20
