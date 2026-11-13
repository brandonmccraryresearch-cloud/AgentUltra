"""Research session management.

Implements the session protocol from the specification:

  1. Read current manuscript
  2. Identify specific task
  3. Select appropriate tools
  4. Execute computation
  5. Verify results
  6. Classify findings
  7. Grade honestly
  8. Update manuscript
  9. Store key facts
  10. Report progress
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agent_ultra.core.config import AgentConfig
from agent_ultra.core.epistemic import EpistemicHygieneChecker, HygieneReport
from agent_ultra.core.failure_detection import FailureDetector
from agent_ultra.core.pipeline import (
    PipelineResult,
    ReasoningPipeline,
)

logger = logging.getLogger(__name__)


class SessionStatus(str, Enum):
    """Status of a research session."""

    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskCategory(str, Enum):
    """Category of the research task."""

    MATH = "math"
    PHYSICS = "physics"
    COMPUTATION = "computation"
    PROOF = "proof"
    EVALUATION = "evaluation"


@dataclass
class QualityGate:
    """A single quality gate check."""

    name: str
    description: str
    passed: bool | None = None
    details: str = ""


@dataclass
class SessionFact:
    """A key fact stored for future session continuity."""

    key: str
    value: str
    category: str = "general"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class SessionLog:
    """A log entry within a session."""

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    phase: str = ""
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchSession:
    """Manages the lifecycle of a single research session."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    status: SessionStatus = SessionStatus.INITIALIZED
    task_description: str = ""
    task_category: TaskCategory = TaskCategory.PHYSICS
    config: AgentConfig = field(default_factory=AgentConfig.default)

    # Artefacts produced during the session
    pipeline_results: list[PipelineResult] = field(default_factory=list)
    hygiene_reports: list[HygieneReport] = field(default_factory=list)
    stored_facts: list[SessionFact] = field(default_factory=list)
    logs: list[SessionLog] = field(default_factory=list)
    quality_gates: list[QualityGate] = field(default_factory=list)

    # Timestamps
    started_at: str | None = None
    completed_at: str | None = None

    def __post_init__(self) -> None:
        self._pipeline = ReasoningPipeline()
        self._hygiene = EpistemicHygieneChecker()
        self._failure_detector = FailureDetector()
        self._initialize_quality_gates()

    # -- Quality gates -------------------------------------------------------

    def _initialize_quality_gates(self) -> None:
        """Set up the standard quality gates from the spec.

        Gate names match ``PipelineResult.passed_quality_gates`` exactly
        wherever the pipeline computes the gate, so session and pipeline
        reports can be aggregated and compared directly. Two gates are
        session-only (the pipeline does not compute them): statistical
        significance and honesty_check.
        """
        gate_definitions = [
            ("dimensional_consistency", "All equations have correct units"),
            ("limiting_cases", "Known physics recovered in appropriate limits"),
            ("numerical_stability", "Result stable under grid resolution changes"),
            ("statistical_significance", "Error bars computed and reported"),
            ("independence_verification", "At least one alternative computation method"),
            ("honesty_check", "No overclaiming; all caveats stated"),
            (
                "classification_assigned",
                "Properly labeled as derivation/prediction/postdiction/calibration",
            ),
            (
                "falsifiability_assessed",
                "Falsifiability of the claim has been assessed",
            ),
            (
                "confidence_graded",
                "An overall confidence grade has been assigned",
            ),
        ]
        self.quality_gates = [
            QualityGate(name=name, description=desc) for name, desc in gate_definitions
        ]

    def update_quality_gate(self, name: str, passed: bool, details: str = "") -> None:
        """Update the status of a quality gate."""
        for gate in self.quality_gates:
            if gate.name == name:
                gate.passed = passed
                gate.details = details
                return
        raise ValueError(f"Unknown quality gate: {name}")

    def all_gates_passed(self) -> bool:
        """Check whether every quality gate has passed."""
        return all(g.passed is True for g in self.quality_gates)

    # -- Session lifecycle ---------------------------------------------------

    def start(self, task: str, category: TaskCategory = TaskCategory.PHYSICS) -> None:
        """Begin the research session."""
        self.task_description = task
        self.task_category = category
        self.status = SessionStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._log("session", f"Session started — task: {task}")

    def evaluate_claim(self, statement: str, **kwargs: Any) -> PipelineResult:
        """Run a claim through the full four-phase pipeline."""
        self._log("pipeline", f"Evaluating claim: {statement[:80]}…")
        result = self._pipeline.run(statement, **kwargs)
        self.pipeline_results.append(result)
        # Sync the pipeline-computed quality gates into the session gates
        # (names match by construction; None means "not assessed" and
        # leaves the session gate untouched).
        for name, passed in result.passed_quality_gates.items():
            if passed is None:
                continue
            for gate in self.quality_gates:
                if gate.name == name:
                    gate.passed = passed
                    break
        return result

    def run_hygiene_audit(self, **kwargs: Any) -> HygieneReport:
        """Run an epistemic hygiene audit."""
        self._log("hygiene", "Running epistemic hygiene audit")
        report = self._hygiene.full_audit(**kwargs)
        self.hygiene_reports.append(report)
        return report

    def store_fact(self, key: str, value: str, category: str = "general") -> None:
        """Store a fact for future session continuity."""
        fact = SessionFact(key=key, value=value, category=category)
        self.stored_facts.append(fact)
        self._log("facts", f"Stored fact: {key}={value}")

    def complete(self) -> None:
        """Mark the session as completed."""
        self.status = SessionStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self._log("session", "Session completed")

    def fail(self, reason: str = "") -> None:
        """Mark the session as failed."""
        self.status = SessionStatus.FAILED
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self._log("session", f"Session failed: {reason}")

    # -- Summary / report ----------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Produce a summary report for the session."""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "task": self.task_description,
            "category": self.task_category.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "claims_evaluated": len(self.pipeline_results),
            "hygiene_audits": len(self.hygiene_reports),
            "quality_gates": {
                g.name: g.passed for g in self.quality_gates
            },
            "all_gates_passed": self.all_gates_passed(),
            "stored_facts": len(self.stored_facts),
        }

    # -- Internal helpers ----------------------------------------------------

    def _log(self, phase: str, message: str, data: dict[str, Any] | None = None) -> None:
        entry = SessionLog(phase=phase, message=message, data=data or {})
        self.logs.append(entry)
        logger.info("[%s] %s", phase, message)
