"""Tests for the research session management."""

import pytest

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
        with pytest.raises(ValueError, match="Unknown quality gate"):
            session.update_quality_gate("nonexistent_gate", True)

    def test_gate_names_align_with_pipeline(self):
        # Session gate names must match PipelineResult.passed_quality_gates
        # keys wherever the pipeline computes the gate.
        session = ResearchSession()
        session.start("test")
        result = session.evaluate_claim("test claim")
        pipeline_gates = result.passed_quality_gates
        session_gates = {g.name: g.passed for g in session.quality_gates}
        for name in pipeline_gates:
            assert name in session_gates, f"pipeline gate {name!r} missing from session"
        # ... and the pipeline-computed values were synced in. A bare
        # pipeline run assigns no classification and assesses no
        # falsifiability, so those gates sync to False (not "unknown").
        assert session_gates["confidence_graded"] is True
        assert session_gates["classification_assigned"] is False
        assert session_gates["falsifiability_assessed"] is False
