"""Basic research example demonstrating Veritas usage.

This example shows how to conduct a simple research query
using the Veritas research agent with Playwright browser control.
"""

import asyncio
import os
from dotenv import load_dotenv

from veritas.core.agent import ResearchAgent, ResearchTask


async def basic_research_example():
    """Run a basic research query about AI regulation."""
    # Load environment variables (for OPENAI_API_KEY)
    load_dotenv()
    
    print("🔍 Veritas Basic Research Example\n")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment.")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("\nContinuing with demo mode (will fail on LLM calls)...\n")
    
    # Initialize the agent
    agent = ResearchAgent(
        model="gpt-4o-mini",  # Using mini for cost efficiency in examples
        max_iterations=2,
        enable_verification=True,
    )
    
    # Create a research task
    task = ResearchTask(
        query="Latest news about AI regulation 2024",
        max_steps=5,
        depth="medium",
    )
    
    print(f"Query: {task.query}")
    print(f"Depth: {task.depth}, Max Steps: {task.max_steps}\n")
    print("-" * 50)
    
    # Execute research
    print("\n🌐 Starting web research...\n")
    
    try:
        result = await agent.research(task)
        
        print("\n" + "=" * 50)
        print("✅ Research Complete!\n")
        
        if result.success:
            print(f"📊 Steps taken: {result.steps_taken}")
            print(f"📚 Sources consulted: {len(result.sources)}\n")
            
            print("📝 Sources:")
            for i, source in enumerate(result.sources, 1):
                print(f"  {i}. {source['title']}")
                print(f"     {source['url']}\n")
            
            print("=" * 50)
            print("💡 ANSWER:\n")
            print(result.answer)
        else:
            print(f"❌ Research failed: {result.error}")
            
    except Exception as e:
        print(f"\n❌ Error during research: {e}")
        print("\nMake sure you have:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Installed Playwright browsers: playwright install chromium")


async def compare_python_versions_example():
    """Another example: comparing Python versions."""
    print("\n" + "=" * 50)
    print("🔍 Example 2: Python Version Comparison\n")
    
    agent = ResearchAgent(model="gpt-4o-mini")
    
    task = ResearchTask(
        query="What are the main differences between Python 3.11 and 3.12?",
        max_steps=4,
        depth="medium",
    )
    
    print(f"Query: {task.query}\n")
    
    try:
        result = await agent.research(task)
        
        if result.success:
            print(f"✅ Found {len(result.sources)} sources\n")
            print("💡 ANSWER:\n")
            print(result.answer[:500] + "..." if len(result.answer) > 500 else result.answer)
        else:
            print(f"❌ Failed: {result.error}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(basic_research_example())
    
    # Uncomment to run second example:
    # asyncio.run(compare_python_versions_example())
