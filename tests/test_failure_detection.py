"""Tests for failure mode detection."""


from agent_ultra.core.failure_detection import (
    ENERGY,
    MOMENTUM,
    DependencyEdge,
    DependencyGraph,
    DependencyNode,
    DimensionSignature,
    FailureDetector,
    FailureMode,
    check_dimensional_consistency,
)


class TestDimensionSignature:
    def test_dimensionless(self):
        d = DimensionSignature()
        assert d.is_dimensionless

    def test_energy(self):
        e = ENERGY
        assert not e.is_dimensionless
        assert e.M == 1 and e.L == 2 and e.T == -2

    def test_multiply(self):
        # force × length = energy
        f = DimensionSignature(M=1, L=1, T=-2)
        length = DimensionSignature(L=1)
        result = f * length
        assert result == ENERGY

    def test_divide(self):
        # energy / momentum = velocity
        result = ENERGY / MOMENTUM
        assert result == DimensionSignature(L=1, T=-1)


class TestDimensionalConsistency:
    def test_consistent(self):
        alert = check_dimensional_consistency(ENERGY, ENERGY)
        assert alert is None

    def test_inconsistent(self):
        alert = check_dimensional_consistency(ENERGY, MOMENTUM)
        assert alert is not None
        assert alert.mode == FailureMode.DIMENSIONAL_INCONSISTENCY


class TestDependencyGraph:
    def test_no_cycles(self):
        g = DependencyGraph()
        g.add_node(DependencyNode("A"))
        g.add_node(DependencyNode("B"))
        g.add_node(DependencyNode("C"))
        g.add_edge(DependencyEdge("A", "B"))
        g.add_edge(DependencyEdge("B", "C"))
        assert g.find_cycles() == []

    def test_cycle_detected(self):
        g = DependencyGraph()
        g.add_node(DependencyNode("A"))
        g.add_node(DependencyNode("B"))
        g.add_node(DependencyNode("C"))
        g.add_edge(DependencyEdge("A", "B"))
        g.add_edge(DependencyEdge("B", "C"))
        g.add_edge(DependencyEdge("C", "A"))
        cycles = g.find_cycles()
        assert len(cycles) > 0


class TestFailureDetector:
    def test_circular_reasoning(self):
        g = DependencyGraph()
        for nid in ["X", "Y"]:
            g.add_node(DependencyNode(nid))
        g.add_edge(DependencyEdge("X", "Y"))
        g.add_edge(DependencyEdge("Y", "X"))

        detector = FailureDetector()
        alerts = detector.detect_circular_reasoning(g)
        assert len(alerts) > 0
        assert alerts[0].mode == FailureMode.CIRCULAR_REASONING

    def test_dimensional_issues(self):
        detector = FailureDetector()
        alerts = detector.detect_dimensional_issues([
            (ENERGY, ENERGY),
            (ENERGY, MOMENTUM),
        ])
        assert len(alerts) == 1
        assert alerts[0].location == "equation_1"

    def test_silent_assumptions(self):
        detector = FailureDetector()
        alerts = detector.detect_silent_assumptions(
            stated_assumptions=[],
            derivation_steps=[
                "We apply the chain rule.",
                "It is obviously true that x > 0.",
                "Clearly, the integral converges.",
            ],
        )
        assert len(alerts) == 2  # "obviously" and "clearly"

    def test_silent_assumptions_suppressed_when_stated(self):
        detector = FailureDetector()
        alerts = detector.detect_silent_assumptions(
            stated_assumptions=["x > 0"],
            derivation_steps=[
                "It is obviously true that x > 0.",
                "Clearly, the integral converges.",
            ],
        )
        # The x > 0 step is stated, not silent; only "clearly" alerts.
        assert len(alerts) == 1
        assert "clearly" in alerts[0].description

    def test_silent_assumptions_suppressed_when_well_known(self):
        detector = FailureDetector()
        alerts = detector.detect_silent_assumptions(
            stated_assumptions=[],
            derivation_steps=[
                "It is obviously true that the integral converges.",
                "Clearly, the integral converges.",
            ],
            well_known_assumptions={"the integral converges"},
        )
        assert len(alerts) == 0

    def test_run_all(self):
        detector = FailureDetector()
        alerts = detector.run_all(
            equations=[(ENERGY, ENERGY)],
            derivation_steps=["Trivially, the result follows."],
        )
        assert len(alerts) >= 1  # at least the "trivially" alert
