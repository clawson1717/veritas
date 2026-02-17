"""Command-line interface for Veritas."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint

from veritas import (
    ResearchAgent,
    ResearchTask,
    ChecklistVerifier,
    get_checklist,
    list_checklists,
    CATTSAllocator,
    AllocationConfig,
    TopologyRouter,
    create_router,
)

console = Console()


def setup_parser() -> argparse.ArgumentParser:
    """Set up the argument parser with all commands and options."""
    parser = argparse.ArgumentParser(
        prog="veritas",
        description="Veritas - An adaptive web research agent with step-by-step verification",
    )
    
    # Global options
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to configuration file (JSON or YAML)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for results"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Research command
    research_parser = subparsers.add_parser(
        "research",
        help="Run a research task"
    )
    research_parser.add_argument(
        "--query", "-q",
        required=True,
        help="Research query"
    )
    research_parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum research iterations"
    )
    
    # Verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify content against a checklist"
    )
    verify_parser.add_argument(
        "--content", "-C",
        required=True,
        help="Content to verify"
    )
    verify_parser.add_argument(
        "--checklist", "-l",
        default="research_quality",
        help="Checklist name to use"
    )
    
    # Allocate command
    allocate_parser = subparsers.add_parser(
        "allocate",
        help="Allocate compute budget based on uncertainty"
    )
    allocate_parser.add_argument(
        "--uncertainty", "-u",
        type=float,
        required=True,
        help="Uncertainty score (0-1)"
    )
    allocate_parser.add_argument(
        "--budget", "-b",
        type=float,
        default=100.0,
        help="Total budget available"
    )
    
    # Route command
    route_parser = subparsers.add_parser(
        "route",
        help="Route task to appropriate agents"
    )
    route_parser.add_argument(
        "--task", "-t",
        required=True,
        help="Task description"
    )
    route_parser.add_argument(
        "--agents", "-a",
        help="Comma-separated list of available agents"
    )
    
    return parser


def load_config(config_path: Optional[Path]) -> dict[str, Any]:
    """Load configuration from file."""
    if config_path is None:
        return {}
    
    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        sys.exit(1)
    
    with open(config_path) as f:
        if config_path.suffix in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(f) or {}
        else:
            return json.load(f)


def run_research(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Run a research task."""
    if args.verbose:
        console.print(f"[cyan]Running research for: {args.query}[/cyan]")
        console.print(f"[cyan]Max iterations: {args.max_iterations}[/cyan]")
    
    try:
        # Create research task
        task = ResearchTask(
            query=args.query,
            max_iterations=args.max_iterations
        )
        
        # Create and run agent
        agent = ResearchAgent(config=config)
        result = agent.run(task)
        
        output_data = {
            "query": result.task.query,
            "answer": result.final_answer,
            "iterations": result.iterations,
            "confidence": result.confidence,
        }
        
        if args.verbose:
            console.print(Panel(
                f"[bold]Query:[/bold] {result.task.query}\n\n"
                f"[bold]Answer:[/bold] {result.final_answer}\n\n"
                f"[bold]Iterations:[/bold] {result.iterations}\n"
                f"[bold]Confidence:[/bold] {result.confidence:.2f}",
                title="Research Result"
            ))
        else:
            console.print(result.final_answer)
        
        # Write output if specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            console.print(f"[green]Results written to {args.output}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error during research: {e}[/red]")
        if args.verbose:
            raise
        sys.exit(1)


def run_verify(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Verify content against a checklist."""
    if args.verbose:
        console.print(f"[cyan]Verifying content against: {args.checklist}[/cyan]")
    
    try:
        # Get the checklist
        checklist = get_checklist(args.checklist)
        
        # Create verifier
        verifier = ChecklistVerifier(config=config)
        
        # Run verification
        result = verifier.verify(args.content, checklist)
        
        output_data = {
            "checklist": args.checklist,
            "content_length": len(args.content),
            "passed": result.passed,
            "score": result.score,
            "results": [
                {
                    "item": item.item,
                    "status": item.status.value,
                    "reason": item.reason
                }
                for item in result.items
            ]
        }
        
        if args.verbose:
            status_color = "green" if result.passed else "red"
            console.print(Panel(
                f"[bold]Checklist:[/bold] {args.checklist}\n"
                f"[bold]Score:[/bold] {result.score:.1%}\n"
                f"[bold]Status:[/bold] [{status_color}]{'PASSED' if result.passed else 'FAILED'}[/{status_color}]",
                title="Verification Result"
            ))
            console.print("\n[bold]Items:[/bold]")
            for item in result.items:
                status_icon = "✓" if item.status.value == "passed" else "✗" if item.status.value == "failed" else "○"
                status_color = "green" if item.status.value == "passed" else "red" if item.status.value == "failed" else "yellow"
                console.print(f"  {status_icon} [{status_color}]{item.item}[/{status_color}]: {item.reason}")
        else:
            console.print(f"Verification: {'PASSED' if result.passed else 'FAILED'} ({result.score:.1%})")
        
        # Write output if specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            console.print(f"[green]Results written to {args.output}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error during verification: {e}[/red]")
        if args.verbose:
            raise
        sys.exit(1)


def run_allocate(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Allocate compute budget based on uncertainty."""
    if args.verbose:
        console.print(f"[cyan]Calculating allocation for uncertainty: {args.uncertainty}[/cyan]")
        console.print(f"[cyan]Budget: {args.budget}[/cyan]")
    
    try:
        # Create allocator
        alloc_config = AllocationConfig(
            initial_budget=args.budget,
            min_budget_fraction=0.1,
            max_budget_fraction=1.0,
        )
        allocator = CATTSAllocator(alloc_config)
        
        # Calculate allocation
        decision = allocator.allocate(args.uncertainty)
        
        output_data = {
            "uncertainty": args.uncertainty,
            "budget": args.budget,
            "allocated": decision.allocated_budget,
            "strategy": decision.strategy.value,
            "reasoning": decision.reasoning
        }
        
        if args.verbose:
            console.print(Panel(
                f"[bold]Uncertainty:[/bold] {args.uncertainty:.2f}\n"
                f"[bold]Budget:[/bold] ${args.budget:.2f}\n"
                f"[bold]Allocated:[/bold] ${decision.allocated_budget:.2f}\n"
                f"[bold]Strategy:[/bold] {decision.strategy.value}\n\n"
                f"[bold]Reasoning:[/bold] {decision.reasoning}",
                title="Allocation Decision"
            ))
        else:
            console.print(f"Allocated: ${decision.allocated_budget:.2f} ({decision.strategy.value})")
        
        # Write output if specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            console.print(f"[green]Results written to {args.output}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error during allocation: {e}[/red]")
        if args.verbose:
            raise
        sys.exit(1)


def run_route(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Route task to appropriate agents."""
    if args.verbose:
        console.print(f"[cyan]Routing task: {args.task}[/cyan]")
    
    try:
        # Create router
        router = create_router(config)
        
        # Parse available agents
        available_agents = args.agents.split(",") if args.agents else None
        
        # Route the task
        result = router.route(args.task, available_agents)
        
        output_data = {
            "task": args.task,
            "recommended_agent": result.recommended_agent,
            "reasoning": result.reasoning,
            "scores": result.agent_scores
        }
        
        if args.verbose:
            console.print(Panel(
                f"[bold]Task:[/bold] {args.task}\n\n"
                f"[bold]Recommended:[/bold] {result.recommended_agent}\n\n"
                f"[bold]Reasoning:[/bold] {result.reasoning}",
                title="Routing Decision"
            ))
            if result.agent_scores:
                console.print("\n[bold]Agent Scores:[/bold]")
                for agent, score in result.agent_scores.items():
                    console.print(f"  {agent}: {score:.2f}")
        else:
            console.print(f"Routed to: {result.recommended_agent}")
        
        # Write output if specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            console.print(f"[green]Results written to {args.output}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error during routing: {e}[/red]")
        if args.verbose:
            raise
        sys.exit(1)


def list_available_checklists() -> None:
    """List all available checklists."""
    checklists = list_checklists()
    console.print("[bold]Available Checklists:[/bold]")
    for name, description in checklists.items():
        console.print(f"  • {name}: {description}")


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args(argv)
    
    # Load config
    config = load_config(args.config)
    
    # Handle no command
    if args.command is None:
        parser.print_help()
        console.print("\n[bold]Available Commands:[/bold]")
        console.print("  research  - Run a research task")
        console.print("  verify    - Verify content against a checklist")
        console.print("  allocate  - Allocate compute budget based on uncertainty")
        console.print("  route     - Route task to appropriate agents")
        
        # Show available checklists
        console.print("\n[bold]Tip:[/bold] Use 'veritas verify --list-checklists' to see available checklists")
        return 0
    
    # Execute command
    try:
        if args.command == "research":
            run_research(args, config)
        elif args.command == "verify":
            run_verify(args, config)
        elif args.command == "allocate":
            run_allocate(args, config)
        elif args.command == "route":
            run_route(args, config)
        else:
            console.print(f"[red]Unknown command: {args.command}[/red]")
            return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if args.verbose:
            raise
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
