# AgentUltra — Autonomous Frontier Mathematics & Physics Research Agent

AgentUltra is a Python framework for autonomous scientific discovery in theoretical mathematics and physics. It implements a rigorous four-phase reasoning pipeline with built-in epistemic hygiene, failure mode detection, and rubric-based evaluation.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Research Session                          │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ Pipeline   │  │  Epistemic   │  │ Failure Detection  │   │
│  │ (4 phases) │  │  Hygiene     │  │ (DAG, dimensions)  │   │
│  └─────┬──────┘  └──────┬───────┘  └─────────┬──────────┘   │
│        │                │                     │              │
│  ┌─────▼──────────────▼───────────────────▼──────────┐      │
│  │              Evaluation & Scoring                   │      │
│  └─────────────────────┬───────────────────────────────┘      │
│                        │                                      │
│  ┌─────────────────────▼───────────────────────────────┐      │
│  │           MCP Server Interfaces                      │      │
│  │  math-mcp │ lean-lsp │ lattice │ group-theory │ data │      │
│  └─────────────────────────────────────────────────────┘      │
│                                                                │
│  ┌──────────────────────┐  ┌─────────────────────────────┐    │
│  │ Symbolic Computation │  │ Numerical Engine             │    │
│  │ (root systems, BZ)   │  │ (MC, FFT, bootstrap, etc.)  │    │
│  └──────────────────────┘  └─────────────────────────────┘    │
│                                                                │
│  ┌─────────────────────────────────────────────────────┐      │
│  │ Knowledge Base (CODATA, PDG, Planck cosmology)       │      │
│  └─────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

## Core Components

### Four-Phase Reasoning Pipeline

Every theoretical claim passes through:

| Phase | Purpose | Key Outputs |
|---|---|---|
| **Comprehension** | Parse structure, identify assumptions, map dependencies | Formal problem statement |
| **Verification** | Formal proof (Lean 4), numerical cross-check, dimensional analysis | Machine-checked proof |
| **Grounding** | Empirical comparison, falsifiability audit | Precision confidence grade |
| **Synthesis** | Framework integration, corollary generation | Updated confidence grade |

### Epistemic Hygiene

Strict protocols prevent overclaiming:

- **DERIVATION** — result follows from axioms without fitted parameters
- **PREDICTION** — model output independent of test data
- **POSTDICTION** — observed value enters derivation chain
- **CALIBRATION** — parameters tuned to match experiment
- **TAUTOLOGY** — algebraic identity, not a physical prediction

### Failure Mode Detection

Automated detection of:
- Circular reasoning (dependency graph cycle detection)
- Dimensional inconsistency (SI unit tracking)
- Silent assumptions (trigger word detection)
- Tautology smuggling (constant passthrough analysis)

### MCP Server Integration

Connects to specialized computation servers via JSON-RPC 2.0:
- `math-mcp` — symbolic algebra
- `lean-lsp-mcp` — formal proof verification
- `lattice-mcp` — lattice field theory
- `group-theory-mcp` — Lie algebra computations
- `experimental-data-mcp` — PDG, CODATA, Planck databases

### Evaluation System

Rubric-based scoring (A/B/C/F) across five criteria:
- Derivation quality
- Verification level
- Precision match
- Falsifiability
- Honesty

With anti-pattern penalties for overclaiming, tautology smuggling, precision inflation, etc.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
# Show configuration
agent-ultra config

# Evaluate a theoretical claim
agent-ultra evaluate "The Schwarzschild metric solves the vacuum Einstein equations"

# Look up a physical constant
agent-ultra constants alpha_inv
```

### Python API

```python
from agent_ultra.core.session import ResearchSession, TaskCategory

# Start a research session
session = ResearchSession()
session.start("Derive electron mass from lattice dynamics", category=TaskCategory.PHYSICS)

# Evaluate a claim through the four-phase pipeline
result = session.evaluate_claim("The electron mass arises from D₄ lattice geometry")

# Run epistemic hygiene audit
report = session.run_hygiene_audit(
    claimed_classification="DERIVATION",
    fitted_parameters=["lattice_spacing"],
)

# Check quality gates
print(result.passed_quality_gates)
session.complete()
print(session.summary())
```

### Symbolic Computation

```python
from agent_ultra.computation.symbolic import construct_root_system

# Construct the D₄ root system (SO(8))
d4 = construct_root_system("D", 4)
print(f"Rank: {d4.rank}, Positive roots: {d4.num_positive_roots}")
print(f"Cartan matrix: {d4.cartan_matrix}")
```

### Numerical Computation

```python
from agent_ultra.computation.numerical import monte_carlo_integrate, bootstrap_error
import numpy as np

# Monte Carlo integration
result = monte_carlo_integrate(
    lambda x: np.exp(-np.sum(x**2, axis=1)),
    lower=np.array([-3.0, -3.0]),
    upper=np.array([3.0, 3.0]),
    samples=1_000_000,
)
print(f"Integral: {result.estimate:.6f} ± {result.std_error:.6f}")
```

## Testing

```bash
python -m pytest tests/ -v
```

## Project Structure

```
src/agent_ultra/
├── core/
│   ├── config.py            # YAML-based agent configuration
│   ├── pipeline.py          # Four-phase reasoning pipeline
│   ├── epistemic.py         # Epistemic hygiene checker
│   ├── failure_detection.py # Failure mode detection
│   └── session.py           # Research session management
├── mcp/
│   ├── client.py            # MCP base client (JSON-RPC 2.0)
│   └── servers.py           # Specialized MCP server interfaces
├── computation/
│   ├── symbolic.py          # Root systems, Brillouin zone integration
│   └── numerical.py         # Monte Carlo, FFT, bootstrap, Richardson
├── evaluation/
│   └── scorer.py            # Rubric-based grading system
├── knowledge/
│   └── constants.py         # CODATA, PDG, Planck data
└── cli/
    └── main.py              # CLI entry point
```

## License

MIT
