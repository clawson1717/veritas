"""Integrated pipeline demo using AgentIntegrator.

This demo shows how to use the full Veritas pipeline with:
- DyTopo routing
- CATTS allocation
- CM2 verification
- iGRPO refinement
"""

import asyncio
from veritas.core.integrator import create_full_integrator


async def main():
    print("=" * 60)
    print("Veritas Integrated Pipeline Demo")
    print("=" * 60)
    
    # Create full integrator with all components
    integrator = create_full_integrator()
    
    # Example queries to demonstrate different scenarios
    queries = [
        {
            "query": "What is quantum computing?",
            "description": "Simple informational query"
        },
        {
            "query": "Analyze the impact of AI on employment",
            "description": "Complex analysis query"
        },
    ]
    
    for i, item in enumerate(queries, 1):
        print(f"\n--- Query {i}: {item['description']} ---")
        print(f"Query: {item['query']}")
        
        # Run the integrated pipeline
        result = await integrator.run(
            query=item["query"],
            context={
                "sources": ["https://example.com"],
                "step_type": "synthesis"
            },
            initial_response=f"Initial answer about: {item['query']}"
        )
        
        print(f"\nResult:")
        print(f"  Success: {result.success}")
        print(f"  Answer: {result.answer[:100]}...")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Steps executed: {result.steps_executed}")
        
        if result.error:
            print(f"  Error: {result.error}")
        
        # Show verification details if available
        if result.verification:
            print(f"  Verification score: {result.verification.score:.2f}")
            print(f"  Passed: {result.verification.passed}")
    
    # Show budget summary
    print("\n--- Budget Summary ---")
    summary = integrator.get_budget_summary()
    print(f"Total allocations: {summary.get('total_allocations', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
