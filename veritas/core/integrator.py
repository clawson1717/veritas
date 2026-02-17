"""Integration layer combining all Veritas components.

This module provides the AgentIntegrator class that combines:
- CATTS Allocator: Dynamic compute allocation based on uncertainty
- CM2 Verification: Checklist-based result verification
- iGRPO Refinement: Self-feedback driven iterative refinement
- DyTopo Router: Dynamic topology routing via semantic matching

The integrator orchestrates the full research pipeline:
1. Route query to appropriate agents (DyTopo)
2. Allocate compute resources (CATTS)
3. Verify results (CM2)
4. Refine answers if needed (iGRPO)
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from veritas.allocation.catts import (
    CATTSAllocator,
    AllocationConfig,
    AllocationDecision,
    AllocationStrategy,
)
from veritas.verification.checklist import (
    ChecklistVerifier,
    ChecklistItem,
    ChecklistResult,
    CheckStatus,
    get_checklist,
)
from veritas.refinement.igrpo import (
    SelfFeedbackRefiner,
    RefinementResult,
    ConvergenceConfig,
)
from veritas.topology.router import (
    TopologyRouter,
    TaskNeeds,
    AgentOffer,
    Complexity,
    Domain,
    create_router,
)
from veritas.core.uncertainty import uncertainty_to_confidence


@dataclass
class IntegrationConfig:
    """Configuration for the AgentIntegrator.
    
    Attributes:
        enable_routing: Whether to use DyTopo routing
        enable_allocation: Whether to use CATTS allocation
        enable_verification: Whether to use CM2 verification
        enable_refinement: Whether to use iGRPO refinement
        verification_checklist: Name of checklist to use ("retrieval", "synthesis", etc.)
        refinement_model: Model to use for refinement
        refinement_temperature: Temperature for refinement generation
        max_refinement_iterations: Maximum refinement iterations
        confidence_threshold: Minimum confidence to skip refinement
    """
    enable_routing: bool = True
    enable_allocation: bool = True
    enable_verification: bool = True
    enable_refinement: bool = True
    verification_checklist: str = "synthesis"
    refinement_model: str = "gpt-4o-mini"
    refinement_temperature: float = 0.7
    max_refinement_iterations: int = 5
    confidence_threshold: float = 0.8


@dataclass
class IntegrationResult:
    """Result of the integrated pipeline.
    
    Attributes:
        success: Whether the pipeline completed successfully
        answer: The final answer (possibly refined)
        confidence: Confidence score (0-1)
        allocation: Compute allocation decision
        verification: Verification result
        refinement: Refinement result
        routed_agents: Agents selected by routing
        steps_executed: List of execution steps taken
        error: Error message if failed
    """
    success: bool
    answer: str
    confidence: float = 0.0
    allocation: Optional[AllocationDecision] = None
    verification: Optional[ChecklistResult] = None
    refinement: Optional[RefinementResult] = None
    routed_agents: list[AgentOffer] = field(default_factory=list)
    steps_executed: list[str] = field(default_factory=list)
    error: Optional[str] = None


class AgentIntegrator:
    """Integrates all Veritas components into a unified pipeline.
    
    This class orchestrates the full research pipeline by combining:
    - DyTopo for intelligent routing
    - CATTS for compute allocation
    - CM2 for verification
    - iGRPO for refinement
    
    The pipeline flow:
    1. Parse the task and determine routing (DyTopo)
    2. Allocate compute budget (CATTS)
    3. Execute the task
    4. Verify the result (CM2)
    5. Refine if confidence is low (iGRPO)
    
    Example:
        >>> integrator = AgentIntegrator()
        >>> result = await integrator.run(
        ...     query="What is quantum computing?",
        ...     context={"sources": ["https://example.com"]}
        ... )
        >>> print(result.answer)
    """
    
    def __init__(
        self,
        config: Optional[IntegrationConfig] = None,
        router: Optional[TopologyRouter] = None,
        allocator: Optional[CATTSAllocator] = None,
        verifier: Optional[ChecklistVerifier] = None,
        refiner: Optional[SelfFeedbackRefiner] = None,
    ):
        """Initialize the AgentIntegrator.
        
        Args:
            config: Integration configuration
            router: Optional pre-configured TopologyRouter
            allocator: Optional pre-configured CATTSAllocator
            verifier: Optional pre-configured ChecklistVerifier
            refiner: Optional pre-configured SelfFeedbackRefiner
        """
        self.config = config or IntegrationConfig()
        
        # Initialize components
        self.router = router or create_router()
        self.allocator = allocator or CATTSAllocator()
        self.verifier = verifier or get_checklist(self.config.verification_checklist)
        
        # Only create refiner if enabled and API key is available
        self.refiner: Optional[SelfFeedbackRefiner] = None
        if self.config.enable_refinement:
            try:
                convergence_config = ConvergenceConfig(
                    max_iterations=self.config.max_refinement_iterations,
                )
                self.refiner = SelfFeedbackRefiner(
                    model=self.config.refinement_model,
                    temperature=self.config.refinement_temperature,
                    convergence_config=convergence_config,
                )
            except Exception:
                # If API key not available, disable refinement
                self.refiner = None
        
        # Execution state
        self._current_query: str = ""
        self._current_context: dict = {}
        self._execution_history: list[IntegrationResult] = []
    
    @property
    def router(self) -> TopologyRouter:
        """Get the topology router."""
        return self._router
    
    @router.setter
    def router(self, value: TopologyRouter) -> None:
        """Set the topology router."""
        self._router = value
    
    @property
    def allocator(self) -> CATTSAllocator:
        """Get the CATTS allocator."""
        return self._allocator
    
    @allocator.setter
    def allocator(self, value: CATTSAllocator) -> None:
        """Set the CATTS allocator."""
        self._allocator = value
    
    @property
    def verifier(self) -> ChecklistVerifier:
        """Get the checklist verifier."""
        return self._verifier
    
    @verifier.setter
    def verifier(self, value: ChecklistVerifier) -> None:
        """Set the checklist verifier."""
        self._verifier = value
    
    async def run(
        self,
        query: str,
        context: dict[str, Any],
        initial_response: Optional[str] = None,
        task_needs: Optional[TaskNeeds] = None,
    ) -> IntegrationResult:
        """Run the full integrated pipeline.
        
        Args:
            query: The research query
            context: Context including sources, previous steps, etc.
            initial_response: Optional initial response to verify and refine
            task_needs: Optional task requirements for routing
            
        Returns:
            IntegrationResult with the final answer and metadata
        """
        self._current_query = query
        self._current_context = context
        
        steps: list[str] = []
        error_msg: Optional[str] = None
        allocation: Optional[AllocationDecision] = None
        verification: Optional[ChecklistResult] = None
        refinement: Optional[RefinementResult] = None
        routed_agents: list[AgentOffer] = []
        
        try:
            # Step 1: Routing (DyTopo)
            if self.config.enable_routing:
                steps.append("routing")
                task_needs = task_needs or self._infer_task_needs(query)
                routed_agents = self.router.route(task_needs)
            
            # Step 2: Allocation (CATTS)
            if self.config.enable_allocation:
                steps.append("allocation")
                vote_distribution = context.get("votes", [])
                step_type = context.get("step_type", "synthesis")
                
                if vote_distribution:
                    allocation = self.allocator.allocate(
                        vote_distribution,
                        step_type=step_type,
                        step_context=context,
                    )
                else:
                    # Pre-allocation based on query complexity
                    initial = self.allocator.allocate_for_generation(query)
                    allocation = AllocationDecision(
                        samples=initial["samples"],
                        token_budget=initial["token_budget"],
                        uncertainty=0.5,
                        confidence=0.5,
                        should_continue=True,
                        reasoning="Initial allocation based on query complexity",
                    )
            
            # Step 3: Verification (CM2)
            answer = initial_response or context.get("answer", "")
            
            if self.config.enable_verification and answer:
                steps.append("verification")
                verification = self.verifier.verify(answer)
            
            # Step 4: Refinement (iGRPO)
            confidence = verification.score if verification else 0.0
            
            if self.config.enable_refinement and answer and self.refiner:
                if confidence < self.config.confidence_threshold:
                    steps.append("refinement")
                    refinement_context = {
                        "question": query,
                        "constraints": context.get("constraints", ""),
                        "sources": context.get("sources", []),
                    }
                    
                    refinement = await self.refiner.refine(
                        initial_response=answer,
                        context=refinement_context,
                    )
                    
                    if refinement.was_improved:
                        answer = refinement.refined
                        
                        # Re-verify after refinement
                        if self.config.enable_verification:
                            verification = self.verifier.verify(answer)
                            confidence = verification.score if verification else 0.0
                else:
                    steps.append("refinement_skipped:high_confidence")
            
            # Final confidence calculation
            final_confidence = confidence
            if refinement:
                final_confidence = max(confidence, refinement.confidence_after)
            
            result = IntegrationResult(
                success=True,
                answer=answer,
                confidence=final_confidence,
                allocation=allocation,
                verification=verification,
                refinement=refinement,
                routed_agents=routed_agents,
                steps_executed=steps,
            )
            
        except Exception as e:
            error_msg = str(e)
            result = IntegrationResult(
                success=False,
                answer=initial_response or context.get("answer", ""),
                error=error_msg,
                steps_executed=steps,
            )
        
        self._execution_history.append(result)
        return result
    
    async def run_simple(
        self,
        query: str,
        answer: str,
    ) -> IntegrationResult:
        """Simplified run for quick verification and refinement.
        
        Args:
            query: The original query
            answer: The answer to process
            
        Returns:
            IntegrationResult
        """
        return await self.run(
            query=query,
            context={"answer": answer},
            initial_response=answer,
        )
    
    def _infer_task_needs(self, query: str) -> TaskNeeds:
        """Infer task needs from query.
        
        Args:
            query: The research query
            
        Returns:
            TaskNeeds with inferred capabilities
        """
        query_lower = query.lower()
        
        # Determine complexity
        complexity = Complexity.MEDIUM
        if any(word in query_lower for word in ["explain", "describe", "what is"]):
            complexity = Complexity.SIMPLE
        elif any(word in query_lower for word in ["analyze", "compare", "evaluate"]):
            complexity = Complexity.COMPLEX
        
        # Determine domain
        domain = Domain.GENERAL
        if any(word in query_lower for word in ["research", "find", "search"]):
            domain = Domain.RESEARCH
        elif any(word in query_lower for word in ["verify", "check", "confirm"]):
            domain = Domain.VERIFICATION
        elif any(word in query_lower for word in ["write", "compose", "create"]):
            domain = Domain.WRITING
        
        # Determine required capabilities
        capabilities = ["information_gathering"]
        if "synthesize" in query_lower or "summarize" in query_lower:
            capabilities.append("summarization")
        if "verify" in query_lower or "check" in query_lower:
            capabilities.append("validation")
        if "write" in query_lower or "create" in query_lower:
            capabilities.append("content_creation")
        
        return TaskNeeds(
            description=query,
            required_capabilities=capabilities,
            complexity=complexity,
            domain=domain,
        )
    
    def reset(self) -> None:
        """Reset the integrator state."""
        self.allocator.reset()
        self.verifier.reset()
        self._current_query = ""
        self._current_context = {}
    
    def get_history(self) -> list[IntegrationResult]:
        """Get execution history."""
        return self._execution_history.copy()
    
    def get_budget_summary(self) -> dict:
        """Get budget usage summary from allocator."""
        return self.allocator.get_budget_summary()


# =============================================================================
# Factory Functions
# =============================================================================

def create_integrator(
    enable_routing: bool = True,
    enable_allocation: bool = True,
    enable_verification: bool = True,
    enable_refinement: bool = True,
    verification_checklist: str = "synthesis",
    **kwargs,
) -> AgentIntegrator:
    """Create a configured AgentIntegrator.
    
    Args:
        enable_routing: Enable DyTopo routing
        enable_allocation: Enable CATTS allocation
        enable_verification: Enable CM2 verification
        enable_refinement: Enable iGRPO refinement
        verification_checklist: Checklist to use for verification
        **kwargs: Additional configuration options
        
    Returns:
        Configured AgentIntegrator
    """
    config = IntegrationConfig(
        enable_routing=enable_routing,
        enable_allocation=enable_allocation,
        enable_verification=enable_verification,
        enable_refinement=enable_refinement,
        verification_checklist=verification_checklist,
        **kwargs,
    )
    return AgentIntegrator(config=config)


def create_minimal_integrator() -> AgentIntegrator:
    """Create a minimal integrator with only verification enabled.
    
    Returns:
        AgentIntegrator with verification only
    """
    return create_integrator(
        enable_routing=False,
        enable_allocation=False,
        enable_verification=True,
        enable_refinement=False,
    )


def create_full_integrator() -> AgentIntegrator:
    """Create a full integrator with all components enabled.
    
    Returns:
        AgentIntegrator with all features
    """
    return create_integrator(
        enable_routing=True,
        enable_allocation=True,
        enable_verification=True,
        enable_refinement=True,
    )


# Export public API
__all__ = [
    "AgentIntegrator",
    "IntegrationConfig",
    "IntegrationResult",
    "create_integrator",
    "create_minimal_integrator",
    "create_full_integrator",
]
