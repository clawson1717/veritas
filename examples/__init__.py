"""Examples for Veritas usage.

Demos:
- basic_research_demo: ResearchAgent with verification
- integrated_pipeline_demo: Full AgentIntegrator pipeline
- catts_demo: CATTS dynamic compute allocation
"""

# Import demos for easy access
from examples.basic_research_demo import main as run_basic_research
from examples.integrated_pipeline_demo import main as run_integrated_pipeline
from examples.catts_demo import main as run_catts_demo

__all__ = [
    "run_basic_research",
    "run_integrated_pipeline", 
    "run_catts_demo",
]
