"""
Example demonstrating CM2 checklist-based verification.

Shows how Veritas uses binary criteria to verify research outputs
and generate actionable feedback for refinement.
"""

from veritas.verification.checklist import ChecklistVerifier, Criterion


def verification_example():
    """Demonstrate CM2 verification."""
    print("✓ CM2 Checklist Verification Example\n")
    
    # Initialize verifier with default criteria
    verifier = ChecklistVerifier()
    
    print("Default verification criteria:")
    for criterion in verifier.criteria:
        print(f"  • [{criterion.id}] {criterion.description}")
        print(f"    Weight: {criterion.weight}")
    
    print(f"\nMinimum pass threshold: {verifier.min_pass_threshold}\n")
    
    # Example answer to verify
    example_answer = """
    Python 3.12 introduces several performance improvements including
    optimized interpreter startup and reduced memory usage. The new
    f-string parsing is more flexible, allowing nested f-strings.
    """
    
    print("Example answer:")
    print(example_answer)
    print("\nVerification result: (placeholder)")
    print("Note: Full implementation pending. This is a structure preview.")


if __name__ == "__main__":
    verification_example()
