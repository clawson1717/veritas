"""Tests for the Veritas CLI."""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from veritas.cli import main, setup_parser, load_config


class TestCLIArgumentParser:
    """Tests for the argument parser setup."""
    
    def test_parser_creation(self):
        """Test that the parser is created successfully."""
        parser = setup_parser()
        assert parser is not None
        assert parser.prog == "veritas"
    
    def test_global_options(self):
        """Test global options are available."""
        parser = setup_parser()
        
        help_output = StringIO()
        with patch("sys.stdout", help_output):
            try:
                parser.parse_args(["--help"])
            except SystemExit:
                pass
        
        help_text = help_output.getvalue()
        assert "--verbose" in help_text or "-v" in help_text
    
    def test_research_command(self):
        """Test research command is available."""
        parser = setup_parser()
        
        args = parser.parse_args(["research", "--query", "What is AI?"])
        assert args.command == "research"
        assert args.query == "What is AI?"
    
    def test_verify_command(self):
        """Test verify command is available."""
        parser = setup_parser()
        
        args = parser.parse_args(["verify", "--content", "test answer"])
        assert args.command == "verify"
        assert args.content == "test answer"
    
    def test_allocate_command(self):
        """Test allocate command is available."""
        parser = setup_parser()
        
        args = parser.parse_args(["allocate", "--uncertainty", "0.5"])
        assert args.command == "allocate"
        assert args.uncertainty == 0.5
    
    def test_route_command(self):
        """Test route command is available."""
        parser = setup_parser()
        
        args = parser.parse_args(["route", "--task", "test task"])
        assert args.command == "route"
        assert args.task == "test task"


class TestCLIMain:
    """Tests for the main CLI function."""
    
    def test_no_command_shows_help(self):
        """Test that no command shows help."""
        with patch("sys.stdout", StringIO()) as mock_stdout:
            with patch("sys.argv", ["veritas"]):
                result = main([])
        
        assert result == 0
    
    def test_invalid_command(self):
        """Test that invalid command shows help."""
        with patch("sys.stdout", StringIO()):
            with patch("sys.argv", ["veritas", "invalid"]):
                result = main([])
        
        assert result == 0
    
    def test_research_requires_query(self):
        """Test that research command requires query."""
        parser = setup_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(["research"])
    
    def test_verify_requires_answer(self):
        """Test that verify command requires answer."""
        parser = setup_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(["verify"])
    
    def test_allocate_requires_votes(self):
        """Test that allocate command requires votes."""
        parser = setup_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(["allocate"])
    
    def test_route_requires_task(self):
        """Test that route command requires task."""
        parser = setup_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(["route"])


class TestLoadConfig:
    """Tests for config loading."""
    
    def test_load_no_config(self):
        """Loading with no config file returns empty dict."""
        config = load_config(None)
        assert config == {}


class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    def test_allocate_command_runs(self):
        """Test allocate command runs without error."""
        from veritas.allocation.catts import CATTSAllocator
        
        allocator = CATTSAllocator()
        decision = allocator.allocate(["click", "click", "type"], step_type="test")
        
        assert decision is not None
        assert decision.samples > 0
    
    def test_route_command_runs(self):
        """Test route command runs without error."""
        from veritas.topology.router import create_router, TaskNeeds, Complexity, Domain
        
        router = create_router()
        task_needs = TaskNeeds(
            description="test query",
            required_capabilities=["info"],
            complexity=Complexity.SIMPLE,
            domain=Domain.GENERAL
        )
        
        result = router.route(task_needs)
        
        assert result is not None


class TestCLIOutput:
    """Tests for CLI output options."""
    
    def test_allocate_output(self):
        """Test allocate produces output."""
        from veritas.allocation.catts import CATTSAllocator
        
        allocator = CATTSAllocator()
        decision = allocator.allocate(["click", "click"], step_type="test")
        
        output = StringIO()
        output.write(f"Samples: {decision.samples}")
        output.write(f"Token budget: {decision.token_budget}")
        
        text = output.getvalue()
        assert "Samples" in text
        assert "Token budget" in text
