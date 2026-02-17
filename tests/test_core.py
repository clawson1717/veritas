"""Tests for Veritas core modules."""

import pytest
from unittest.mock import patch, MagicMock
from veritas.core.agent import ResearchAgent, ResearchTask


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


class TestResearchAgent:
    """Test suite for ResearchAgent."""
    
    @patch('veritas.core.agent.AsyncOpenAI')
    def test_agent_initialization(self, mock_openai):
        """Test agent can be initialized."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        agent = ResearchAgent()
        assert agent.model == "gpt-4o-mini"
        assert agent.max_iterations == 3
        assert agent.enable_verification is True
    
    @patch('veritas.core.agent.AsyncOpenAI')
    def test_agent_custom_config(self, mock_openai):
        """Test agent with custom configuration."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        agent = ResearchAgent(
            model="gpt-4o",
            max_iterations=5,
            enable_verification=False,
        )
        assert agent.model == "gpt-4o"
        assert agent.max_iterations == 5
        assert agent.enable_verification is False
