"""Integrated Pipeline Demo.

Demonstrates the full Veritas pipeline using AgentIntegrator:
- Route: Dynamic topology routing to select agents
- Allocate: CATTS dynamic compute allocation  
- Verify: CM2 checklist-based verification
- Refine: iGRPO self-feedback refinement
"""

import asyncio
from veritas.core.integrator import (
    AgentIntegrator,
    IntegrationConfig,
    create_full_integrator,
    create_minimal_integrator,
)
from veritas.topology.router import TaskNeeds, Complexity, Domain
from veritas.verification.checklist import get_checklist


async def main():
    """Run the integrated pipeline demo."""
    print("=" * 60)
    print("Veritas Integrated Pipeline Demo")
    print("=" * 60)
    
    # Demo 1: Full integrator with all features
    print("\n[1] Full Integrator Demo")
    print("-" * 40)
    
    integrator = create_full_integrator()
    print(f"✓ Created full integrator")
    print(f"  - Routing: {integrator.config.enable_routing}")
    print(f"  - Allocation: {integrator.config.enable_allocation}")
    print(f"  - Verification: {integrator.config.enable_verification}")
    print(f"  - Refinement: {integrator.config.enable_refinement}")
    
    # Test routing
    print("\n--- Testing Routing ---")
    task_needs = TaskNeeds(
        description="Research the benefits of renewable energy",
        required_capabilities=["information_gathering", "summarization", "validation"],
        complexity=Complexity.COMPLEX,
        domain=Domain.RESEARCH
    )
    
    routed_agents = integrator.router.route(task_needs)
    print(f"Routed agents for task:")
    for agent in routed_agents:
        print(f"  • {agent.name} ({agent.agent_id}): {agent.capabilities[:3]}...")
    
    # Test allocation
    print("\n--- Testing CATTS Allocation ---")
    
    # Simulate vote distributions with varying uncertainty
    vote_scenarios = [
        (["A", "A", "A"], "High agreement (certain)"),
        (["A", "B", "C"], "No agreement (uncertain)"),
        (["A", "A", "B"], "Moderate agreement"),
    ]
    
    for votes, description in vote_scenarios:
        decision = integrator.allocator.allocate(votes, step_type="synthesis")
        print(f"\nVotes: {votes}")
        print(f"  {description}")
        print(f"  → Uncertainty: {decision.uncertainty:.2f}")
        print(f"  → Confidence: {decision.confidence:.2f}")
        print(f"  → Tokens: {decision.token_budget}")
        print(f"  → Samples: {decision.samples}")
        print(f"  → Reasoning: {decision.reasoning}")
    
    # Demo 2: Minimal integrator with verification only
    print("\n" + "=" * 60)
    print("[2] Minimal Integrator Demo")
    print("-" * 40)
    
    minimal = create_minimal_integrator()
    print(f"✓ Created minimal integrator")
    print(f"  - Verification only: {minimal.config.enable_verification}")
    
    # Test verification pipeline
    print("\n--- Testing Verification Pipeline ---")
    
    test_answers = [
        {
            "query": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
        },
        {
            "query": "What is machine learning?",
            "answer": "IDK."
        }
    ]
    
    for test in test_answers:
        result = await minimal.run_simple(test["query"], test["answer"])
        print(f"\nQuery: {test['query']}")
        print(f"Answer: {test['answer'][:50]}...")
        print(f"Success: {result.success}")
        print(f"Confidence: {result.confidence:.2f}")
        if result.verification:
            print(f"Verification: {result.verification.score:.1%} ({result.verification.passed}/{result.verification.total} checks)")
    
    # Demo 3: Full pipeline execution
    print("\n" + "=" * 60)
    print("[3] Full Pipeline Execution")
    print("-" * 40)
    
    # Create fresh integrator
    full = create_full_integrator()
    
    # Simulate a research scenario with context
    query = "What are the main advantages of solar energy?"
    initial_answer = "Solar energy is renewable and sustainable."
    context = {
        "answer": initial_answer,
        "sources": [
            {"title": "Solar Energy Benefits", "url": "https://example.com/1"},
            {"title": "Renewable Energy Guide", "url": "https://example.com/2"},
        ],
        "step_type": "synthesis",
        "constraints": "Provide scientific backing"
    }
    
    print(f"\nRunning full pipeline...")
    print(f"Query: {query}")
    print(f"Initial answer: {initial_answer}")
    
    # This will run: route -> allocate -> verify -> (refine if needed)
    result = await full.run(
        query=query,
        context=context,
        initial_response=initial_answer
    )
    
    print(f"\n--- Pipeline Results ---")
    print(f"Success: {result.success}")
    print(f"Steps executed: {result.steps_executed}")
    print(f"Confidence: {result.confidence:.2f}")
    
    if result.allocation:
        print(f"\nAllocation:")
        print(f"  - Tokens: {result.allocation.token_budget}")
        print(f"  - Samples: {result.allocation.samples}")
        print(f"  - Uncertainty: {result.allocation.uncertainty:.2f}")
    
    if result.verification:
        print(f"\nVerification:")
        print(f"  - Score: {result.verification.score:.1%}")
        print(f"  - Passed: {result.verification.passed}/{result.verification.total}")
        print(f"  - All checks passed: {result.verification.all_passed}")
    
    if result.refinement:
        print(f"\nRefinement:")
        print(f"  - Was improved: {result.refinement.was_improved}")
        print(f"  - Confidence after: {result.refinement.confidence_after:.2f}")
    
    # Demo 4: Budget summary
    print("\n" + "=" * 60)
    print("[4] Budget Summary")
    print("-" * 40)
    
    summary = full.get_budget_summary()
    print(f"Total tokens used: {summary.get('total_tokens', 0)}")
    print(f"Steps processed: {summary.get('steps', 0)}")
    
    savings = summary.get('savings_vs_uniform', {})
    if savings:
        print(f"Savings vs uniform: {savings.get('savings', 0)} tokens ({savings.get('percent', 0)}%)")


if __name__ == "__main__":
    asyncio.run(main())
