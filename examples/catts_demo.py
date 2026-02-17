"""CATTS dynamic compute allocation demo.

Demonstrates how CATTS (Context-Aware Test-Time Scaling) 
allocates compute resources based on uncertainty.
"""

from veritas.allocation.catts import CATTSAllocator


def main():
    print("=" * 60)
    print("CATTS Dynamic Compute Allocation Demo")
    print("=" * 60)
    
    # Create CATTS allocator
    allocator = CATTSAllocator()
    
    # Simulate vote distributions with different uncertainties
    scenarios = [
        {
            "name": "High Consensus",
            "votes": ["click", "click", "click"]
        },
        {
            "name": "Medium Uncertainty",
            "votes": ["click", "type", "click"]
        },
        {
            "name": "High Disagreement",
            "votes": ["click", "navigate", "extract"]
        },
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        # Allocate compute
        decision = allocator.allocate(
            scenario["votes"],
            step_type="action_selection"
        )
        
        print(f"  Samples: {decision.samples}")
        print(f"  Token budget: {decision.token_budget}")
        print(f"  Uncertainty: {decision.uncertainty:.2f}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Should continue: {decision.should_continue}")
        print(f"  Reasoning: {decision.reasoning}")
    
    # Show budget summary
    print("\n--- Budget Summary ---")
    summary = allocator.get_budget_summary()
    print(f"Total allocations: {summary.get('total_allocations', 0)}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
