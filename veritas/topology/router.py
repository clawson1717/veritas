"""Dynamic topology router implementing DyTopo routing.

Reconstructs agent communication graphs each round via semantic
matching of agent "needs" and "offers".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
from enum import Enum


class Complexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class Domain(Enum):
    """Task domains."""
    GENERAL = "general"
    RESEARCH = "research"
    VERIFICATION = "verification"
    WRITING = "writing"
    ANALYSIS = "analysis"
    TECHNICAL = "technical"


@dataclass
class TaskNeeds:
    """Represents the requirements of a task.
    
    Attributes:
        description: Natural language description of the task
        required_capabilities: List of capabilities needed to complete the task
        complexity: Complexity level of the task
        domain: Domain the task belongs to
    """
    description: str
    required_capabilities: list[str]
    complexity: Complexity = Complexity.MEDIUM
    domain: Domain = Domain.GENERAL


@dataclass
class AgentOffer:
    """Represents an agent's offered capabilities.
    
    Attributes:
        agent_id: Unique identifier for the agent
        name: Human-readable name for the agent
        capabilities: List of capabilities the agent provides
        specialization: Primary specialization domain
    """
    agent_id: str
    name: str
    capabilities: list[str]
    specialization: str = "general"


@dataclass
class AgentCapability:
    """An agent's offered or needed capability."""
    name: str
    description: str
    type: str  # "offer" or "need"
    embedding: list[float] | None = None


@dataclass  
class AgentNode:
    """A node in the dynamic topology graph."""
    agent_id: str
    capabilities: list[AgentCapability]
    handler: Callable


class AgentRegistry:
    """Maintains a registry of available agents and their capabilities.
    
    Provides methods to register, lookup, and query agents based on
    their capabilities.
    """
    
    def __init__(self):
        self._agents: dict[str, AgentOffer] = {}
        self._capability_index: dict[str, list[str]] = {}  # capability -> agent_ids
    
    def register(self, agent: AgentOffer) -> None:
        """Register an agent with the registry.
        
        Args:
            agent: The agent offer to register
        """
        self._agents[agent.agent_id] = agent
        
        # Update capability index
        for cap in agent.capabilities:
            cap_lower = cap.lower()
            if cap_lower not in self._capability_index:
                self._capability_index[cap_lower] = []
            if agent.agent_id not in self._capability_index[cap_lower]:
                self._capability_index[cap_lower].append(agent.agent_id)
    
    def get_agent(self, agent_id: str) -> AgentOffer | None:
        """Get an agent by ID.
        
        Args:
            agent_id: The agent's unique identifier
            
        Returns:
            The agent offer if found, None otherwise
        """
        return self._agents.get(agent_id)
    
    def find_agents_by_capability(self, capability: str) -> list[AgentOffer]:
        """Find agents that offer a specific Args:
            capability capability.
        
       : The capability to search for
            
        Returns:
            List of agents offering the capability
        """
        cap_lower = capability.lower()
        agent_ids = self._capability_index.get(cap_lower, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    def list_agents(self) -> list[AgentOffer]:
        """List all registered agents.
        
        Returns:
            List of all registered agent offers
        """
        return list(self._agents.values())
    
    def get_all_capabilities(self) -> set[str]:
        """Get all unique capabilities across all agents.
        
        Returns:
            Set of all available capabilities
        """
        return set(self._capability_index.keys())


class TopologyRouter:
    """Dynamic topology router via semantic matching.
    
    Inspired by DyTopo, this router:
    1. Maintains a registry of agent capabilities
    2. Each round, matches "needs" to "offers" via semantic similarity
    3. Reconstructs the communication graph dynamically
    4. Routes subtasks to appropriate agents
    
    Uses keyword matching for similarity when embeddings are not available.
    
    Args:
        similarity_threshold: Minimum similarity for matching (0.0 to 1.0)
        embedding_fn: Optional function to compute semantic embeddings
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.5,
        embedding_fn: Callable[[str], list[float]] | None = None,
    ):
        self.similarity_threshold = similarity_threshold
        self._embedding_fn = embedding_fn
        self._registry = AgentRegistry()
        self._graph: dict[str, list[str]] = {}  # agent_id -> connected_agents
        
        # Register default agent profiles
        self._register_default_agents()
    
    def _register_default_agents(self) -> None:
        """Register the pre-built agent profiles."""
        default_agents = [
            AgentOffer(
                agent_id="researcher",
                name="Research Agent",
                capabilities=["search", "web_scraping", "information_gathering", "web_search", "data_collection"],
                specialization="research"
            ),
            AgentOffer(
                agent_id="verifier",
                name="Verification Agent",
                capabilities=["fact_checking", "citation_verification", "quality_check", "validation", "verification"],
                specialization="verification"
            ),
            AgentOffer(
                agent_id="synthesizer",
                name="Synthesis Agent",
                capabilities=["summary", "writing", "content_creation", "summarization", "composition"],
                specialization="writing"
            ),
            AgentOffer(
                agent_id="refiner",
                name="Refinement Agent",
                capabilities=["editing", "improvement", "iteration", "refinement", "polishing"],
                specialization="refinement"
            ),
        ]
        
        for agent in default_agents:
            self._registry.register(agent)
    
    @property
    def registry(self) -> AgentRegistry:
        """Get the agent registry."""
        return self._registry
    
    def register_agent(self, agent: AgentOffer) -> None:
        """Register a custom agent with the router.
        
        Args:
            agent: The agent offer to register
        """
        self._registry.register(agent)
    
    def _compute_keyword_similarity(
        self,
        needed: str,
        offered: str,
    ) -> float:
        """Compute similarity between capability keywords.
        
        Uses simple keyword matching: counts matching words and
        computes Jaccard-like similarity.
        
        Args:
            needed: The needed capability
            offered: The offered capability
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        needed_words = set(needed.lower().replace("_", " ").split())
        offered_words = set(offered.lower().replace("_", " ").split())
        
        if not needed_words or not offered_words:
            return 0.0
        
        intersection = needed_words & offered_words
        union = needed_words | offered_words
        
        # Jaccard similarity
        return len(intersection) / len(union) if union else 0.0
    
    def _compute_embedding_similarity(
        self,
        needed: str,
        offered: str,
    ) -> float:
        """Compute similarity using embeddings if available.
        
        Args:
            needed: The needed capability description
            offered: The offered capability description
            
        Returns:
            Cosine similarity between embeddings, or 0.0 if embeddings unavailable
        """
        if self._embedding_fn is None:
            return 0.0
        
        try:
            emb_needed = self._embedding_fn(needed)
            emb_offered = self._embedding_fn(offered)
            
            # Cosine similarity
            dot = sum(a * b for a, b in zip(emb_needed, emb_offered))
            norm_needed = sum(a * a for a in emb_needed) ** 0.5
            norm_offered = sum(a * a for a in emb_offered) ** 0.5
            
            if norm_needed == 0 or norm_offered == 0:
                return 0.0
            
            return dot / (norm_needed * norm_offered)
        except Exception:
            return 0.0
    
    def _compute_similarity(
        self,
        needed: str,
        offered: str,
    ) -> float:
        """Compute overall similarity between capabilities.
        
        Combines embedding similarity (if available) with keyword matching.
        
        Args:
            needed: The needed capability
            offered: The offered capability
            
        Returns:
            Combined similarity score between 0.0 and 1.0
        """
        # Try embedding similarity first
        emb_sim = self._compute_embedding_similarity(needed, offered)
        
        # If embeddings available and gave good result, use it
        if self._embedding_fn is not None and emb_sim > 0:
            return emb_sim
        
        # Otherwise use keyword matching
        return self._compute_keyword_similarity(needed, offered)
    
    def route(self, task_needs: TaskNeeds) -> list[AgentOffer]:
        """Route a task to appropriate agents based on needs.
        
        Matches task requirements to agent capabilities via
        semantic similarity, returning the best-fit agents.
        
        Args:
            task_needs: The task requirements
            
        Returns:
            List of agents best suited for this task, sorted by relevance
        """
        agent_scores: dict[str, float] = {}
        
        # Score each agent based on capability match
        for agent in self._registry.list_agents():
            max_score = 0.0
            
            for needed_cap in task_needs.required_capabilities:
                for offered_cap in agent.capabilities:
                    score = self._compute_similarity(needed_cap, offered_cap)
                    max_score = max(max_score, score)
            
            if max_score >= self.similarity_threshold:
                agent_scores[agent.agent_id] = max_score
        
        # Sort by score descending
        sorted_agents = sorted(
            agent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            self._registry.get_agent(agent_id)
            for agent_id, score in sorted_agents
            if self._registry.get_agent(agent_id) is not None
        ]
    
    def build_team(self, task_needs: TaskNeeds) -> list[AgentOffer]:
        """Assemble a team of agents for a task.
        
        Builds a complementary team by selecting agents that together
        cover all required capabilities.
        
        Args:
            task_needs: The task requirements
            
        Returns:
            List of agents forming the team, in execution order
        """
        if not task_needs.required_capabilities:
            return []
        
        team: list[AgentOffer] = []
        remaining_capabilities = set(
            cap.lower() for cap in task_needs.required_capabilities
        )
        
        # Greedy selection: always pick best agent for remaining needs
        max_iterations = len(task_needs.required_capabilities) * 2
        iterations = 0
        
        while remaining_capabilities and iterations < max_iterations:
            iterations += 1
            
            best_agent = None
            best_agent_caps = set()
            best_score = 0.0
            
            for agent in self._registry.list_agents():
                if agent in team:
                    continue
                
                agent_caps = set(
                    cap.lower() for cap in agent.capabilities
                )
                
                # Count how many remaining capabilities this agent covers
                covered = remaining_capabilities & agent_caps
                
                if covered:
                    # Score based on coverage and capability count
                    coverage_ratio = len(covered) / len(remaining_capabilities)
                    score = coverage_ratio * (1.0 + 0.1 * len(covered))
                    
                    if score > best_score:
                        best_agent = agent
                        best_agent_caps = covered
                        best_score = score
            
            if best_agent is None:
                break
            
            team.append(best_agent)
            remaining_capabilities -= best_agent_caps
        
        return team
    
    def rebuild_topology(self) -> dict[str, list[str]]:
        """Rebuild the communication graph from current capabilities.
        
        Creates connections between agents based on capability 
        complementarity (agents that can feed into each other).
        
        Returns:
            Adjacency list representing agent connections
        """
        graph: dict[str, list[str]] = {}
        agents = self._registry.list_agents()
        
        for agent in agents:
            graph[agent.agent_id] = []
            
            for other in agents:
                if other.agent_id == agent.agent_id:
                    continue
                
                # Check if other agent's capabilities complement this one
                agent_caps = set(cap.lower() for cap in agent.capabilities)
                other_caps = set(cap.lower() for cap in other.capabilities)
                
                # Complementary: one's outputs could be another's inputs
                # Simple heuristic: different capability sets are complementary
                overlap = agent_caps & other_caps
                
                if len(overlap) < len(agent_caps) * 0.5:
                    # Significant difference in capabilities = potential connection
                    graph[agent.agent_id].append(other.agent_id)
        
        self._graph = graph
        return graph
    
    def get_topology(self) -> dict[str, list[str]]:
        """Get the current topology graph.
        
        Returns:
            Adjacency list of agent connections
        """
        if not self._graph:
            self.rebuild_topology()
        return self._graph


# Backwards compatibility alias
class DynamicRouter(TopologyRouter):
    """Backwards compatibility alias for TopologyRouter."""
    pass
