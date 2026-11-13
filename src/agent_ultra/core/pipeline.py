"""Four-phase reasoning pipeline.

Implements the core cognitive architecture described in the specification:

    Phase 1 — COMPREHENSION : Parse structure, identify assumptions, map dependencies.
    Phase 2 — VERIFICATION  : Formal proof, numerical cross-check, dimensional analysis.
    Phase 3 — GROUNDING     : Empirical comparison, falsifiability, classification.
    Phase 4 — SYNTHESIS      : Framework integration, corollary generation, grading.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ClaimType(str, Enum):
    """Classification of a theoretical component."""

    AXIOM = "axiom"
    DEFINITION = "definition"
    THEOREM = "theorem"
    CONJECTURE = "conjecture"
    HEURISTIC = "heuristic"


class ResultClassification(str, Enum):
    """Epistemic classification for a research result."""

    DERIVATION = "DERIVATION"
    PREDICTION = "PREDICTION"
    POSTDICTION = "POSTDICTION"
    CALIBRATION = "CALIBRATION"
    TAUTOLOGY = "TAUTOLOGY"


class VerificationStatus(str, Enum):
    """Outcome of the verification phase."""

    VERIFIED = "VERIFIED"
    FALSIFIED = "FALSIFIED"
    UNRESOLVED = "UNRESOLVED"
    GAP_IDENTIFIED = "GAP_IDENTIFIED"


class ConfidenceGrade(str, Enum):
    """Overall confidence grade for a research finding."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Assumption:
    """A single stated or implicit assumption."""

    description: str
    explicit: bool = True
    source: str = ""


@dataclass
class Dependency:
    """A dependency edge in the reasoning graph."""

    source_claim: str
    target_claim: str
    relationship: str = "depends_on"


@dataclass
class ComprehensionResult:
    """Output of Phase 1 — Comprehension."""

    claim_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    statement: str = ""
    claim_type: ClaimType = ClaimType.CONJECTURE
    assumptions: list[Assumption] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    variables: dict[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass
class VerificationResult:
    """Output of Phase 2 — Verification."""

    status: VerificationStatus = VerificationStatus.UNRESOLVED
    formal_proof: str | None = None
    numerical_checks: list[dict[str, Any]] = field(default_factory=list)
    dimensional_consistent: bool | None = None
    limiting_cases_recovered: bool | None = None
    alternative_derivations: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class GroundingResult:
    """Output of Phase 3 — Grounding."""

    classification: ResultClassification = ResultClassification.DERIVATION
    experimental_comparison: dict[str, Any] = field(default_factory=dict)
    precision_percent: float | None = None
    calibration_inputs: list[str] = field(default_factory=list)
    falsifiable: bool | None = None
    falsifiability_description: str = ""
    notes: str = ""


@dataclass
class SynthesisResult:
    """Output of Phase 4 — Synthesis."""

    confidence_grade: ConfidenceGrade = ConfidenceGrade.C
    consequences: list[str] = field(default_factory=list)
    new_predictions: list[str] = field(default_factory=list)
    parsimony_ratio: float | None = None
    framework_updates: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class PipelineResult:
    """Aggregate result of running a claim through all four phases."""

    claim_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    comprehension: ComprehensionResult | None = None
    verification: VerificationResult | None = None
    grounding: GroundingResult | None = None
    synthesis: SynthesisResult | None = None

    @property
    def passed_quality_gates(self) -> dict[str, bool | None]:
        """Evaluate quality gate checklist."""
        v = self.verification
        g = self.grounding
        s = self.synthesis
        numerical_stability: bool | None = None
        if v:
            numerical_checks = getattr(v, "numerical_checks", None)
            if isinstance(numerical_checks, bool):
                numerical_stability = numerical_checks
            elif isinstance(numerical_checks, dict):
                if isinstance(numerical_checks.get("numerical_stability"), bool):
                    numerical_stability = numerical_checks["numerical_stability"]
                elif isinstance(numerical_checks.get("stable"), bool):
                    numerical_stability = numerical_checks["stable"]
                elif isinstance(numerical_checks.get("passed"), bool):
                    numerical_stability = numerical_checks["passed"]

        return {
            "dimensional_consistency": v.dimensional_consistent if v else None,
            "limiting_cases": v.limiting_cases_recovered if v else None,
            "numerical_stability": numerical_stability,
            "independence_verification": (
                len(v.alternative_derivations) > 0 if v else None
            ),
            "classification_assigned": g.classification is not None if g else None,
            "falsifiability_assessed": g.falsifiable is not None if g else None,
            "confidence_graded": s is not None,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ReasoningPipeline:
    """Four-phase reasoning pipeline for evaluating theoretical claims.

    Each phase can be executed independently or chained via :pymeth:`run`.
    Phases delegate to external tools (MCP servers, proof assistants, numerical
    engines) through callback hooks supplied at construction time.
    """

    def __init__(
        self,
        *,
        comprehension_hook: Any | None = None,
        verification_hook: Any | None = None,
        grounding_hook: Any | None = None,
        synthesis_hook: Any | None = None,
    ) -> None:
        self._comprehension_hook = comprehension_hook
        self._verification_hook = verification_hook
        self._grounding_hook = grounding_hook
        self._synthesis_hook = synthesis_hook

    # -- Phase implementations -----------------------------------------------

    def comprehend(self, statement: str, **kwargs: Any) -> ComprehensionResult:
        """Phase 1 — parse the mathematical structure completely."""
        logger.info("Phase 1 (Comprehension): parsing claim")
        result = ComprehensionResult(statement=statement)
        if self._comprehension_hook is not None:
            result = self._comprehension_hook(result, **kwargs)
        return result

    def verify(
        self, comprehension: ComprehensionResult, **kwargs: Any
    ) -> VerificationResult:
        """Phase 2 — formal proof, numerical cross-check, dimensional analysis."""
        logger.info("Phase 2 (Verification): checking claim %s", comprehension.claim_id)
        result = VerificationResult()
        if self._verification_hook is not None:
            result = self._verification_hook(comprehension, result, **kwargs)
        return result

    def ground(
        self,
        comprehension: ComprehensionResult,
        verification: VerificationResult,
        **kwargs: Any,
    ) -> GroundingResult:
        """Phase 3 — confront theory with empirical data."""
        logger.info("Phase 3 (Grounding): assessing claim %s", comprehension.claim_id)
        result = GroundingResult()
        if self._grounding_hook is not None:
            result = self._grounding_hook(comprehension, verification, result, **kwargs)
        return result

    def synthesize(
        self,
        comprehension: ComprehensionResult,
        verification: VerificationResult,
        grounding: GroundingResult,
        **kwargs: Any,
    ) -> SynthesisResult:
        """Phase 4 — integrate into framework and grade overall confidence."""
        logger.info("Phase 4 (Synthesis): integrating claim %s", comprehension.claim_id)
        result = SynthesisResult()

        # Automatic confidence grading based on earlier phases
        result.confidence_grade = self._auto_grade(verification, grounding)

        if self._synthesis_hook is not None:
            result = self._synthesis_hook(
                comprehension, verification, grounding, result, **kwargs
            )
        return result

    # -- Full pipeline -------------------------------------------------------

    def run(self, statement: str, **kwargs: Any) -> PipelineResult:
        """Execute all four phases sequentially and return the aggregate result."""
        comp = self.comprehend(statement, **kwargs)
        ver = self.verify(comp, **kwargs)
        gnd = self.ground(comp, ver, **kwargs)
        syn = self.synthesize(comp, ver, gnd, **kwargs)
        return PipelineResult(
            claim_id=comp.claim_id,
            comprehension=comp,
            verification=ver,
            grounding=gnd,
            synthesis=syn,
        )

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _auto_grade(
        verification: VerificationResult,
        grounding: GroundingResult,
    ) -> ConfidenceGrade:
        """Heuristic auto-grading based on verification and grounding results."""
        score = 0

        # Verification contribution
        if verification.status == VerificationStatus.VERIFIED:
            score += 3
        elif verification.status == VerificationStatus.GAP_IDENTIFIED:
            score += 1

        if verification.dimensional_consistent:
            score += 1
        if verification.limiting_cases_recovered:
            score += 1
        if len(verification.alternative_derivations) > 0:
            score += 1

        # Grounding contribution
        if grounding.classification == ResultClassification.DERIVATION:
            score += 2
        elif grounding.classification == ResultClassification.PREDICTION:
            score += 2
        elif grounding.classification == ResultClassification.POSTDICTION:
            score += 1

        if grounding.falsifiable:
            score += 1

        precision = grounding.precision_percent
        if precision is not None:
            if precision < 0.1:
                score += 3
            elif precision < 1.0:
                score += 2
            elif precision < 10.0:
                score += 1

        # Map score to grade
        if score >= 10:
            return ConfidenceGrade.A
        if score >= 7:
            return ConfidenceGrade.B
        if score >= 4:
            return ConfidenceGrade.C
        if score >= 2:
            return ConfidenceGrade.D
        return ConfidenceGrade.F
