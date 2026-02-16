# Veritas 🔍

> *An adaptive web research agent that verifies as it goes.*

Veritas is an intelligent research agent that combines cutting-edge techniques from recent AI research to conduct thorough, verified web research. Unlike simple search-and-summarize tools, Veritas **verifies after every retrieval** and iteratively refines its answers using self-feedback.

## ✨ Features

- **🔍 Adaptive Research**: Dynamically allocates compute based on uncertainty (CATTS)
- **✓ Step-by-Step Verification**: Binary checklist verification after every retrieval (CM2 + LawThinker)
- **🔄 Iterative Refinement**: Self-feedback-driven answer improvement (iGRPO)
- **🌐 Smart Routing**: Semantic query routing to optimal information sources (DyTopo)
- **🤖 Browser Automation**: Full Playwright integration for real web interaction

## 🚀 Quick Start

```python
import asyncio
from veritas import ResearchAgent
from veritas.core.agent import ResearchQuery

async def main():
    agent = ResearchAgent()
    
    query = ResearchQuery(
        question="What are the latest developments in quantum error correction?",
        depth=3,
        max_steps=10
    )
    
    result = await agent.research(query)
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Sources: {len(result.sources)}")

asyncio.run(main())
```

## 🏗️ Architecture

Veritas implements five key research insights:

| Component | Paper | Purpose |
|-----------|-------|---------|
| **CATTS Allocator** | Context-Aware Test-Time Scaling | Dynamic compute allocation based on uncertainty |
| **CM2 Verifier** | Checklist-based Multi-step verification | Binary criteria for reliable verification |
| **iGRPO Loop** | Iterative GRPO with self-feedback | Self-improving answer refinement |
| **LawThinker-style** | LawThinker | Verify after every retrieval step |
| **DyTopo Router** | Dynamic Topology | Semantic query-to-capability matching |

## 📦 Installation

```bash
pip install veritas

# Or from source
git clone https://github.com/yourusername/veritas
cd veritas
pip install -e .
```

## 🧪 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run examples
python examples/basic_research.py
```

## 📝 Example Usage

See the `examples/` directory for complete examples:

- `basic_research.py` - Simple research query
- `verified_research.py` - Research with detailed verification
- `adaptive_compute.py` - Demonstrates CATTS allocation

## 📄 License

MIT License - see LICENSE file for details.

## 🔬 Research Acknowledgments

Veritas builds upon recent advances in:
- Test-time compute scaling (CATTS)
- Structured verification (CM2)
- Self-improving agents (iGRPO)
- Retrieval verification (LawThinker)
- Dynamic routing (DyTopo)

---

*Built with ❤️ for better research.*
