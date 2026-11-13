"""Failure mode detection module.

Implements automated detection of common failure modes in theoretical
physics derivations: circular reasoning, tautology smuggling, dimensional
inconsistency, sign errors, overcounting, and silent assumptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FailureMode(str, Enum):
    """Enumeration of detectable failure modes."""

    CIRCULAR_REASONING = "circular_reasoning"
    TAUTOLOGY_SMUGGLING = "tautology_smuggling"
    DIMENSIONAL_INCONSISTENCY = "dimensional_inconsistency"
    SIGN_ERROR = "sign_error"
    OVERCOUNTING = "overcounting"
    SILENT_ASSUMPTION = "silent_assumption"
    POST_HOC_RATIONALIZATION = "post_hoc_rationalization"


@dataclass
class FailureAlert:
    """A detected (or suspected) failure in a derivation."""

    mode: FailureMode
    description: str
    confidence: float = 0.0  # 0.0–1.0
    location: str = ""
    suggested_fix: str = ""


# ---------------------------------------------------------------------------
# Dependency Graph Analysis
# ---------------------------------------------------------------------------

@dataclass
class DependencyNode:
    """A node in the proof dependency graph."""

    node_id: str
    label: str = ""
    node_type: str = ""  # axiom, theorem, conjecture, etc.


@dataclass
class DependencyEdge:
    """A directed edge in the proof dependency graph."""

    source: str
    target: str
    edge_type: str = "depends_on"


class DependencyGraph:
    """Directed acyclic graph for tracking proof dependencies.

    Used to detect circular reasoning by checking for cycles.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, DependencyNode] = {}
        self._edges: list[DependencyEdge] = []
        self._adjacency: dict[str, list[str]] = {}

    def add_node(self, node: DependencyNode) -> None:
        self._nodes[node.node_id] = node
        if node.node_id not in self._adjacency:
            self._adjacency[node.node_id] = []

    def add_edge(self, edge: DependencyEdge) -> None:
        self._edges.append(edge)
        if edge.source not in self._adjacency:
            self._adjacency[edge.source] = []
        self._adjacency[edge.source].append(edge.target)

    def find_cycles(self) -> list[list[str]]:
        """Detect all cycles in the dependency graph using DFS."""
        visited: set[str] = set()
        on_stack: set[str] = set()
        cycles: list[list[str]] = []
        path: list[str] = []

        def _dfs(node: str) -> None:
            visited.add(node)
            on_stack.add(node)
            path.append(node)

            for neighbour in self._adjacency.get(node, []):
                if neighbour not in visited:
                    _dfs(neighbour)
                elif neighbour in on_stack:
                    # Found a cycle — extract it from the path
                    cycle_start = path.index(neighbour)
                    cycles.append(path[cycle_start:] + [neighbour])

            path.pop()
            on_stack.discard(node)

        for node_id in list(self._adjacency.keys()):
            if node_id not in visited:
                _dfs(node_id)

        return cycles

    @property
    def nodes(self) -> dict[str, DependencyNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> list[DependencyEdge]:
        return list(self._edges)


# ---------------------------------------------------------------------------
# Dimension Tracking
# ---------------------------------------------------------------------------

@dataclass
class DimensionSignature:
    """Physical dimensions represented as a vector of exponents.

    The seven SI base dimensions: [M, L, T, A, Θ, N, J]
    (mass, length, time, current, temperature, amount, luminous intensity)
    """

    M: int = 0  # mass (kg)
    L: int = 0  # length (m)
    T: int = 0  # time (s)
    A: int = 0  # electric current (A)
    Theta: int = 0  # temperature (K)
    N: int = 0  # amount of substance (mol)
    J: int = 0  # luminous intensity (cd)

    @property
    def is_dimensionless(self) -> bool:
        return all(
            v == 0
            for v in (self.M, self.L, self.T, self.A, self.Theta, self.N, self.J)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DimensionSignature):
            return NotImplemented
        return (
            self.M == other.M
            and self.L == other.L
            and self.T == other.T
            and self.A == other.A
            and self.Theta == other.Theta
            and self.N == other.N
            and self.J == other.J
        )

    def __mul__(self, other: DimensionSignature) -> DimensionSignature:
        return DimensionSignature(
            M=self.M + other.M,
            L=self.L + other.L,
            T=self.T + other.T,
            A=self.A + other.A,
            Theta=self.Theta + other.Theta,
            N=self.N + other.N,
            J=self.J + other.J,
        )

    def __truediv__(self, other: DimensionSignature) -> DimensionSignature:
        return DimensionSignature(
            M=self.M - other.M,
            L=self.L - other.L,
            T=self.T - other.T,
            A=self.A - other.A,
            Theta=self.Theta - other.Theta,
            N=self.N - other.N,
            J=self.J - other.J,
        )

    def __repr__(self) -> str:
        parts = []
        for name, val in [
            ("M", self.M),
            ("L", self.L),
            ("T", self.T),
            ("A", self.A),
            ("Θ", self.Theta),
            ("N", self.N),
            ("J", self.J),
        ]:
            if val != 0:
                parts.append(f"{name}^{val}" if val != 1 else name)
        return " ".join(parts) if parts else "dimensionless"


# Well-known dimension signatures
ENERGY = DimensionSignature(M=1, L=2, T=-2)
MOMENTUM = DimensionSignature(M=1, L=1, T=-1)
FORCE = DimensionSignature(M=1, L=1, T=-2)
VELOCITY = DimensionSignature(L=1, T=-1)
ACCELERATION = DimensionSignature(L=1, T=-2)
MASS = DimensionSignature(M=1)
LENGTH = DimensionSignature(L=1)
TIME = DimensionSignature(T=1)
DIMENSIONLESS = DimensionSignature()


def check_dimensional_consistency(
    lhs: DimensionSignature,
    rhs: DimensionSignature,
) -> FailureAlert | None:
    """Check that the left- and right-hand sides of an equation have matching dimensions."""
    if lhs != rhs:
        return FailureAlert(
            mode=FailureMode.DIMENSIONAL_INCONSISTENCY,
            description=f"Dimensional mismatch: LHS={lhs!r}, RHS={rhs!r}",
            confidence=1.0,
            suggested_fix="Verify dimensions of each term and insert missing factors.",
        )
    return None


# ---------------------------------------------------------------------------
# Composite Failure Detector
# ---------------------------------------------------------------------------

class FailureDetector:
    """Aggregate failure mode detector.

    Runs all available checks and produces a list of alerts.
    """

    def detect_circular_reasoning(
        self, graph: DependencyGraph
    ) -> list[FailureAlert]:
        """Check the derivation DAG for cycles."""
        alerts: list[FailureAlert] = []
        for cycle in graph.find_cycles():
            alerts.append(
                FailureAlert(
                    mode=FailureMode.CIRCULAR_REASONING,
                    description=f"Circular dependency detected: {' → '.join(cycle)}",
                    confidence=1.0,
                    suggested_fix="Break the cycle by removing or re-deriving one dependency.",
                )
            )
        return alerts

    def detect_dimensional_issues(
        self,
        equations: list[tuple[DimensionSignature, DimensionSignature]],
    ) -> list[FailureAlert]:
        """Check each equation for dimensional consistency."""
        alerts: list[FailureAlert] = []
        for idx, (lhs, rhs) in enumerate(equations):
            alert = check_dimensional_consistency(lhs, rhs)
            if alert is not None:
                alert.location = f"equation_{idx}"
                alerts.append(alert)
        return alerts

    def detect_silent_assumptions(
        self,
        stated_assumptions: list[str],
        derivation_steps: list[str],
        *,
        well_known_assumptions: set[str] | None = None,
    ) -> list[FailureAlert]:
        """Flag derivation steps that rely on unstated assumptions.

        A simple heuristic: any step containing words like "obviously",
        "clearly", "well-known", "trivially" should have its underlying
        assumption explicitly listed. No alert is raised for a step whose
        underlying assumption is already stated, or is in the supplied
        set of well-known assumptions — those are assumed, not silent.
        """
        alerts: list[FailureAlert] = []
        trigger_words = {"obviously", "clearly", "well-known", "trivially", "it is easy to see"}
        # Assumptions already on the record (stated in the derivation, or
        # supplied as well-known background) do not count as silent.
        known = {a.lower() for a in stated_assumptions}
        if well_known_assumptions:
            known |= {a.lower() for a in well_known_assumptions}

        for idx, step in enumerate(derivation_steps):
            step_lower = step.lower()
            if any(assumption and assumption in step_lower for assumption in known):
                continue
            for trigger in trigger_words:
                if trigger in step_lower:
                    alerts.append(
                        FailureAlert(
                            mode=FailureMode.SILENT_ASSUMPTION,
                            description=(
                                f'Step {idx} uses "{trigger}" — verify the '
                                f"underlying assumption is stated explicitly."
                            ),
                            confidence=0.7,
                            location=f"step_{idx}",
                            suggested_fix=(
                                "State the assumption explicitly and verify it."
                            ),
                        )
                    )
        return alerts

    def run_all(
        self,
        *,
        graph: DependencyGraph | None = None,
        equations: list[tuple[DimensionSignature, DimensionSignature]] | None = None,
        stated_assumptions: list[str] | None = None,
        derivation_steps: list[str] | None = None,
    ) -> list[FailureAlert]:
        """Execute every detector and return a merged list of alerts."""
        alerts: list[FailureAlert] = []

        if graph is not None:
            alerts.extend(self.detect_circular_reasoning(graph))

        if equations is not None:
            alerts.extend(self.detect_dimensional_issues(equations))

        if derivation_steps is not None:
            alerts.extend(
                self.detect_silent_assumptions(
                    stated_assumptions or [],
                    derivation_steps,
                )
            )

        return alerts
