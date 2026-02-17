"""Basic Research Agent Demo.

Demonstrates creating a ResearchAgent, running a simple research query,
and basic verification using the Veritas framework.
"""

import asyncio
import os
from veritas.agents.researcher import ResearchTask
from veritas.verification.checklist import get_checklist


async def main():
    """Run the basic research demo."""
    print("=" * 60)
    print("Veritas Basic Research Agent Demo")
    print("=" * 60)
    
    # Check for API key
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    
    if has_api_key:
        from veritas.agents.researcher import ResearchAgent
        
        # Create a research agent
        agent = ResearchAgent(
            model="gpt-4o-mini",
            max_iterations=3,
            enable_verification=True
        )
        print(f"\n✓ Created ResearchAgent (model: {agent.model})")
    else:
        print("\n⚠ No OPENAI_API_KEY found - demonstrating without LLM")
        print("  (Set OPENAI_API_KEY to enable full research)")
    
    # Define a research task
    task = ResearchTask(
        query="What is quantum computing?",
        max_steps=5,
        depth="medium"
    )
    print(f"\n✓ Created ResearchTask: '{task.query}'")
    print(f"  - Max steps: {task.max_steps}")
    print(f"  - Depth: {task.depth}")
    
    # Create a verification checklist
    verifier = get_checklist("synthesis")
    print(f"\n✓ Created verification checklist: '{verifier.name}'")
    print(f"  - Items: {len(verifier.items)}")
    for item in verifier.items:
        print(f"    • {item.id}: {item.description}")
    
    # Run the research task (if API key available)
    print("\n" + "-" * 60)
    if has_api_key:
        print("Running research task...")
        print("-" * 60)
        
        try:
            result = await agent.research(task)
            
            print(f"\nResults:")
            print(f"  - Query: {result.query}")
            print(f"  - Success: {result.success}")
            print(f"  - Steps taken: {result.steps_taken}")
            print(f"  - Sources: {len(result.sources)}")
            
            if result.sources:
                print(f"\n  Top sources:")
                for i, src in enumerate(result.sources[:3], 1):
                    print(f"    {i}. {src.get('title', 'Untitled')}")
                    print(f"       {src.get('url', 'No URL')}")
            
            if result.answer:
                # Verify the answer
                verification = verifier.verify(result.answer)
                print(f"\n  Verification score: {verification.score:.1%}")
                print(f"  Checks passed: {verification.passed}/{verification.total}")
            
            if result.error:
                print(f"\n  Error: {result.error}")
                
        except Exception as e:
            print(f"\n⚠ Could not complete research task: {e}")
    else:
        print("Skipping full research (no API key)")
    
    # Demonstrate verification independently
    print("\n" + "=" * 60)
    print("Demonstrating Verification")
    print("=" * 60)
    
    # Test with sample answers
    sample_answers = [
        "Quantum computing is a type of computation whose operations can exploit phenomena like superposition, interference, and entanglement.",
        "I don't have enough information to answer that question.",
        "Based on research from IBM and Google, quantum computing leverages quantum bits (qubits) that can exist in multiple states simultaneously."
    ]
    
    for answer in sample_answers:
        result = verifier.verify(answer)
        status = "✓" if result.all_passed else "✗"
        print(f"\n{status} Answer: {answer[:50]}...")
        print(f"  Score: {result.score:.1%} ({result.passed}/{result.total} checks)")
    
    # Show checklist feedback
    print("\n" + "=" * 60)
    print("Verification Feedback Example")
    print("=" * 60)
    print(verifier.get_feedback())


if __name__ == "__main__":
    asyncio.run(main())
