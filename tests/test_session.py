"""Tests for the research session management."""

from agent_ultra.core.session import (
    ResearchSession,
    SessionStatus,
    TaskCategory,
)


class TestResearchSession:
    def test_session_lifecycle(self):
        session = ResearchSession()
        assert session.status == SessionStatus.INITIALIZED

        session.start("Derive the Schwarzschild metric")
        assert session.status == SessionStatus.IN_PROGRESS
        assert session.started_at is not None

        result = session.evaluate_claim("Schwarzschild metric solves vacuum EFE")
        assert result is not None
        assert len(session.pipeline_results) == 1

        session.complete()
        assert session.status == SessionStatus.COMPLETED
        assert session.completed_at is not None

    def test_quality_gates(self):
        session = ResearchSession()
        session.start("test")

        # Initially no gates are passed
        assert not session.all_gates_passed()

        # Pass all gates
        for gate in session.quality_gates:
            session.update_quality_gate(gate.name, True)
        assert session.all_gates_passed()

    def test_store_fact(self):
        session = ResearchSession()
        session.start("test")
        session.store_fact("alpha_inv", "137.036", category="constants")
        assert len(session.stored_facts) == 1
        assert session.stored_facts[0].key == "alpha_inv"

    def test_hygiene_audit(self):
        session = ResearchSession()
        session.start("test")
        report = session.run_hygiene_audit(
            claimed_classification="DERIVATION",
            fitted_parameters=["g_s"],
        )
        assert len(report.violations) > 0

    def test_summary(self):
        session = ResearchSession()
        session.start("test", category=TaskCategory.MATH)
        summary = session.summary()
        assert summary["status"] == "in_progress"
        assert summary["category"] == "math"

    def test_fail(self):
        session = ResearchSession()
        session.start("test")
        session.fail("computation diverged")
        assert session.status == SessionStatus.FAILED

    def test_update_unknown_gate_raises(self):
        session = ResearchSession()
        session.start("test")
        try:
            session.update_quality_gate("nonexistent_gate", True)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
