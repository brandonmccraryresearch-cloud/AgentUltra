"""Evaluation and scoring system.

Implements the rubric-based grading system from the specification for
assessing theoretical claims along five criteria:

  1. Derivation quality
  2. Verification level
  3. Precision match
  4. Falsifiability
  5. Honesty
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RubricGrade(str, Enum):
    """Grade for a single rubric criterion."""

    A = "A"
    B = "B"
    C = "C"
    F = "F"


# Points awarded per grade
GRADE_POINTS: dict[RubricGrade, int] = {
    RubricGrade.A: 3,
    RubricGrade.B: 2,
    RubricGrade.C: 1,
    RubricGrade.F: 0,
}


@dataclass
class CriterionScore:
    """Score for a single rubric criterion."""

    criterion: str
    grade: RubricGrade
    points: int = 0
    justification: str = ""

    def __post_init__(self) -> None:
        self.points = GRADE_POINTS[self.grade]


@dataclass
class AntiPatternPenalty:
    """Penalty for a detected anti-pattern."""

    name: str
    description: str
    penalty_points: int = 0


# Predefined anti-patterns from the specification
ANTI_PATTERNS: dict[str, int] = {
    "overclaiming": 3,
    "tautology_smuggling": 3,
    "look_elsewhere_blindness": 2,
    "precision_inflation": 3,
    "silent_calibration": 3,
}


@dataclass
class ScoreCard:
    """Complete evaluation scorecard for a theoretical claim."""

    claim_id: str = ""
    criteria: list[CriterionScore] = field(default_factory=list)
    penalties: list[AntiPatternPenalty] = field(default_factory=list)

    @property
    def raw_score(self) -> int:
        """Sum of criterion points (before penalties)."""
        return sum(c.points for c in self.criteria)

    @property
    def total_penalty(self) -> int:
        """Sum of penalty points."""
        return sum(p.penalty_points for p in self.penalties)

    @property
    def final_score(self) -> int:
        """Net score after penalties (floored at 0)."""
        return max(0, self.raw_score - self.total_penalty)

    @property
    def max_possible(self) -> int:
        """Maximum possible score (3 points × number of criteria)."""
        return len(self.criteria) * 3

    @property
    def overall_grade(self) -> RubricGrade:
        """Map the final score to an overall grade."""
        if self.max_possible == 0:
            return RubricGrade.F
        ratio = self.final_score / self.max_possible
        if ratio >= 0.8:
            return RubricGrade.A
        if ratio >= 0.5:
            return RubricGrade.B
        if ratio >= 0.25:
            return RubricGrade.C
        return RubricGrade.F


class Scorer:
    """Rubric-based scorer for theoretical claims.

    Grades each claim along five criteria and applies anti-pattern penalties.
    """

    CRITERIA_NAMES = [
        "derivation",
        "verification",
        "precision",
        "falsifiability",
        "honesty",
    ]

    def grade_derivation(
        self,
        *,
        from_axioms: bool = False,
        sketch_with_gaps: bool = False,
        motivated_conjecture: bool = False,
    ) -> CriterionScore:
        """Grade the derivation quality."""
        if from_axioms:
            grade = RubricGrade.A
            justification = "Complete derivation from axioms"
        elif sketch_with_gaps:
            grade = RubricGrade.B
            justification = "Sketch with identified gaps"
        elif motivated_conjecture:
            grade = RubricGrade.C
            justification = "Motivated conjecture"
        else:
            grade = RubricGrade.F
            justification = "Ad hoc assertion"
        return CriterionScore(
            criterion="derivation", grade=grade, justification=justification
        )

    def grade_verification(
        self,
        *,
        formal_proof: bool = False,
        numerical_check: bool = False,
        dimensional_check: bool = False,
    ) -> CriterionScore:
        """Grade the verification level."""
        if formal_proof and numerical_check:
            grade = RubricGrade.A
            justification = "Lean 4 proof + numerical cross-check"
        elif numerical_check:
            grade = RubricGrade.B
            justification = "Numerical verification only"
        elif dimensional_check:
            grade = RubricGrade.C
            justification = "Dimensional check only"
        else:
            grade = RubricGrade.F
            justification = "No verification"
        return CriterionScore(
            criterion="verification", grade=grade, justification=justification
        )

    def grade_precision(
        self, *, deviation_percent: float | None = None
    ) -> CriterionScore:
        """Grade precision match with experiment."""
        if deviation_percent is None:
            return CriterionScore(
                criterion="precision",
                grade=RubricGrade.F,
                justification="No experimental comparison",
            )
        if deviation_percent < 0.1:
            grade = RubricGrade.A
            justification = f"<0.1% deviation ({deviation_percent:.4f}%)"
        elif deviation_percent < 1.0:
            grade = RubricGrade.B
            justification = f"<1% deviation ({deviation_percent:.2f}%)"
        elif deviation_percent < 10.0:
            grade = RubricGrade.C
            justification = f"<10% deviation ({deviation_percent:.1f}%)"
        else:
            grade = RubricGrade.F
            justification = f">10% deviation ({deviation_percent:.1f}%) or wrong sign"
        return CriterionScore(
            criterion="precision", grade=grade, justification=justification
        )

    def grade_falsifiability(
        self,
        *,
        precise_quantitative: bool = False,
        semi_quantitative: bool = False,
        qualitative_only: bool = False,
    ) -> CriterionScore:
        """Grade falsifiability of the prediction."""
        if precise_quantitative:
            grade = RubricGrade.A
            justification = "Precise quantitative prediction"
        elif semi_quantitative:
            grade = RubricGrade.B
            justification = "Semi-quantitative prediction"
        elif qualitative_only:
            grade = RubricGrade.C
            justification = "Qualitative prediction only"
        else:
            grade = RubricGrade.F
            justification = "Unfalsifiable"
        return CriterionScore(
            criterion="falsifiability", grade=grade, justification=justification
        )

    def grade_honesty(
        self,
        *,
        all_caveats_stated: bool = False,
        most_caveats_stated: bool = False,
        some_caveats_stated: bool = False,
    ) -> CriterionScore:
        """Grade the honesty of the assessment."""
        if all_caveats_stated:
            grade = RubricGrade.A
            justification = "All caveats and limitations stated"
        elif most_caveats_stated:
            grade = RubricGrade.B
            justification = "Most caveats stated"
        elif some_caveats_stated:
            grade = RubricGrade.C
            justification = "Some caveats stated"
        else:
            grade = RubricGrade.F
            justification = "Overclaiming — caveats not stated"
        return CriterionScore(
            criterion="honesty", grade=grade, justification=justification
        )

    def apply_anti_pattern(
        self, name: str, description: str = ""
    ) -> AntiPatternPenalty:
        """Create a penalty for a detected anti-pattern."""
        points = ANTI_PATTERNS.get(name, 0)
        return AntiPatternPenalty(
            name=name,
            description=description or name,
            penalty_points=points,
        )

    def score(
        self,
        claim_id: str,
        *,
        derivation_kwargs: dict[str, Any] | None = None,
        verification_kwargs: dict[str, Any] | None = None,
        precision_kwargs: dict[str, Any] | None = None,
        falsifiability_kwargs: dict[str, Any] | None = None,
        honesty_kwargs: dict[str, Any] | None = None,
        anti_patterns: list[str] | None = None,
    ) -> ScoreCard:
        """Produce a complete scorecard for a claim."""
        card = ScoreCard(claim_id=claim_id)

        card.criteria.append(self.grade_derivation(**(derivation_kwargs or {})))
        card.criteria.append(self.grade_verification(**(verification_kwargs or {})))
        card.criteria.append(self.grade_precision(**(precision_kwargs or {})))
        card.criteria.append(self.grade_falsifiability(**(falsifiability_kwargs or {})))
        card.criteria.append(self.grade_honesty(**(honesty_kwargs or {})))

        for ap_name in (anti_patterns or []):
            card.penalties.append(self.apply_anti_pattern(ap_name))

        return card
