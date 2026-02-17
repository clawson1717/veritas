"""Command-line interface for Veritas."""

import argparse
import asyncio
import json
import sys
from typing import Optional

from veritas.core.integrator import create_full_integrator, create_minimal_integrator
from veritas.allocation.catts import CATTSAllocator
from veritas.verification.checklist import get_checklist
from veritas.topology.router import create_router


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="veritas",
        description="Veritas - Adaptive research agent with verification"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Research command
    research_parser = subparsers.add_parser("research", help="Run research query")
    research_parser.add_argument("query", help="Research question")
    research_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    research_parser.add_argument("--format", "-f", choices=["json", "text"], default="text", help="Output format")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify an answer")
    verify_parser.add_argument("answer", help="Answer to verify")
    verify_parser.add_argument("--checklist", "-c", default="synthesis", help="Checklist to use")
    verify_parser.add_argument("--format", "-f", choices=["json", "text"], default="text", help="Output format")
    
    # Allocate command
    allocate_parser = subparsers.add_parser("allocate", help="Allocate compute budget")
    allocate_parser.add_argument("votes", nargs="+", help="Vote distribution (e.g., A B A)")
    allocate_parser.add_argument("--step-type", "-s", default="action_selection", help="Step type")
    
    # Route command
    route_parser = subparsers.add_parser("route", help="Route a query")
    route_parser.add_argument("query", help="Query to route")
    route_parser.add_argument("--sources", "-s", nargs="+", default=["web", "database"], help="Available sources")
    
    return parser


async def run_research(args) -> int:
    """Run a research query."""
    integrator = create_full_integrator() if args.verbose else create_minimal_integrator()
    
    result = await integrator.run(
        query=args.query,
        context={"sources": ["web"]},
        initial_response=f"Research on: {args.query}"
    )
    
    if args.format == "json":
        print(json.dumps({
            "success": result.success,
            "answer": result.answer,
            "confidence": result.confidence,
            "steps": result.steps_executed,
            "error": result.error
        }, indent=2))
    else:
        print(f"Query: {args.query}")
        print(f"Success: {result.success}")
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Steps: {', '.join(result.steps_executed)}")
        if result.error:
            print(f"Error: {result.error}")
    
    return 0 if result.success else 1


def run_verify(args) -> int:
    """Verify an answer."""
    verifier = get_checklist(args.checklist)
    result = verifier.verify(args.answer)
    
    if args.format == "json":
        print(json.dumps({
            "passed": result.passed,
            "score": result.score,
            "items": [{"criterion": i.criterion, "status": i.status.value} for i in result.items]
        }, indent=2))
    else:
        print(f"Checklist: {args.checklist}")
        print(f"Passed: {result.passed}")
        print(f"Score: {result.score:.2f}")
        print("\nItems:")
        for item in result.items:
            status = "✓" if item.status.value == "passed" else "✗"
            print(f"  {status} {item.criterion}: {item.status.value}")
    
    return 0 if result.passed else 1


def run_allocate(args) -> int:
    """Allocate compute budget."""
    allocator = CATTSAllocator()
    decision = allocator.allocate(args.votes, step_type=args.step_type)
    
    print(f"Votes: {' '.join(args.votes)}")
    print(f"Step type: {args.step_type}")
    print(f"\nAllocation decision:")
    print(f"  Samples: {decision.samples}")
    print(f"  Token budget: {decision.token_budget}")
    print(f"  Uncertainty: {decision.uncertainty:.2f}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Should continue: {decision.should_continue}")
    print(f"  Reasoning: {decision.reasoning}")
    
    return 0


def run_route(args) -> int:
    """Route a query."""
    router = create_router()
    
    # Simple routing based on keywords
    from veritas.topology.router import TaskNeeds, Complexity, Domain
    
    query_lower = args.query.lower()
    domain = Domain.RESEARCH
    if "verify" in query_lower or "check" in query_lower:
        domain = Domain.VERIFICATION
    
    task_needs = TaskNeeds(
        description=args.query,
        required_capabilities=["information_gathering"],
        complexity=Complexity.MEDIUM,
        domain=domain
    )
    
    agents = router.route(task_needs)
    
    print(f"Query: {args.query}")
    print(f"Available sources: {', '.join(args.sources)}")
    print(f"\nRouted agents: {len(agents)}")
    for agent in agents:
        print(f"  - {agent.name}: {agent.offer[:50]}...")
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "research":
        return asyncio.run(run_research(args))
    elif args.command == "verify":
        return run_verify(args)
    elif args.command == "allocate":
        return run_allocate(args)
    elif args.command == "route":
        return run_route(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
