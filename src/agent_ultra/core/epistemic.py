"""Epistemic hygiene module.

Implements strict epistemic discipline protocols to prevent overclaiming,
tautology smuggling, and misclassification of research results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ViolationType(str, Enum):
    """Types of epistemic hygiene violations."""

    FIT_AS_DERIVATION = "fit_claimed_as_derivation"
    POSTDICTION_AS_PREDICTION = "postdiction_claimed_as_prediction"
    TAUTOLOGY_AS_PREDICTION = "tautology_claimed_as_prediction"
    LOOK_ELSEWHERE_OMITTED = "look_elsewhere_effect_not_reported"
    SILENT_CALIBRATION = "calibration_input_not_disclosed"
    PRECISION_INFLATION = "precision_inflated"
    BARE_INTEGER_CONFUSION = "bare_integer_confused_with_correction"


@dataclass
class HygieneViolation:
    """Record of a detected epistemic hygiene violation."""

    violation_type: ViolationType
    description: str
    severity: int = 0  # 0=info, 1=warning, 2=error, 3=critical
    penalty_points: int = 0
    source_location: str = ""


@dataclass
class CalibrationRecord:
    """Tracks a parameter that was calibrated (fitted) to data."""

    parameter_name: str
    fitted_value: Any = None
    experimental_source: str = ""
    uncertainty: float | None = None


@dataclass
class SearchSpaceRecord:
    """Records the search space explored when finding a formula match."""

    total_formulas_searched: int = 0
    matches_found: int = 0
    search_description: str = ""
    look_elsewhere_factor: float | None = None


@dataclass
class HygieneReport:
    """Full epistemic hygiene assessment for a research claim."""

    violations: list[HygieneViolation] = field(default_factory=list)
    calibrations: list[CalibrationRecord] = field(default_factory=list)
    search_spaces: list[SearchSpaceRecord] = field(default_factory=list)
    is_clean: bool = True
    total_penalty: int = 0

    def add_violation(self, violation: HygieneViolation) -> None:
        """Register a violation and update aggregate state."""
        self.violations.append(violation)
        self.total_penalty += violation.penalty_points
        if violation.severity >= 2:
            self.is_clean = False


class EpistemicHygieneChecker:
    """Enforces epistemic discipline on research claims.

    Rules enforced (from specification):
    1. Never claim a derivation that is actually a fit.
    2. Never claim a prediction that uses observed values as input.
    3. Never confuse algebraic identities with physical predictions.
    4. Always report the look-elsewhere effect.
    5. Always distinguish the bare integer from the correction.
    """

    # -- Public API ----------------------------------------------------------

    def check_derivation_vs_fit(
        self,
        *,
        claimed_classification: str,
        fitted_parameters: list[str] | None = None,
        experimental_inputs: list[str] | None = None,
    ) -> list[HygieneViolation]:
        """Rule 1: Verify that a claimed derivation is not actually a fit."""
        violations: list[HygieneViolation] = []
        fitted_parameters = fitted_parameters or []
        experimental_inputs = experimental_inputs or []

        if claimed_classification.upper() == "DERIVATION" and len(fitted_parameters) > 0:
            violations.append(
                HygieneViolation(
                    violation_type=ViolationType.FIT_AS_DERIVATION,
                    description=(
                        f"Claimed DERIVATION but parameters were fitted: "
                        f"{', '.join(fitted_parameters)}. Should be CALIBRATION."
                    ),
                    severity=3,
                    penalty_points=3,
                )
            )

        if (
            claimed_classification.upper() == "PREDICTION"
            and len(experimental_inputs) > 0
        ):
            violations.append(
                HygieneViolation(
                    violation_type=ViolationType.POSTDICTION_AS_PREDICTION,
                    description=(
                        f"Claimed PREDICTION but observed values used as input: "
                        f"{', '.join(experimental_inputs)}. Should be POSTDICTION."
                    ),
                    severity=3,
                    penalty_points=3,
                )
            )

        return violations

    def check_tautology(
        self,
        *,
        constants_in: set[str] | None = None,
        constants_out: set[str] | None = None,
        is_algebraic_identity: bool = False,
    ) -> list[HygieneViolation]:
        """Rule 3: Detect tautological derivations.

        If fundamental constants (c, ℏ, G, etc.) enter and exit without
        transformation, or if the result is an algebraic identity, flag it.
        """
        violations: list[HygieneViolation] = []
        constants_in = constants_in or set()
        constants_out = constants_out or set()

        passthrough = constants_in & constants_out
        if passthrough:
            violations.append(
                HygieneViolation(
                    violation_type=ViolationType.TAUTOLOGY_AS_PREDICTION,
                    description=(
                        f"Constants enter and exit without transformation: "
                        f"{', '.join(sorted(passthrough))}. May be tautological."
                    ),
                    severity=2,
                    penalty_points=3,
                )
            )

        if is_algebraic_identity:
            violations.append(
                HygieneViolation(
                    violation_type=ViolationType.TAUTOLOGY_AS_PREDICTION,
                    description=(
                        "Result is an algebraic identity — holds for any "
                        "parameter value, not a physical prediction."
                    ),
                    severity=3,
                    penalty_points=3,
                )
            )

        return violations

    def check_look_elsewhere(
        self,
        *,
        search_space_size: int | None = None,
        reported: bool = False,
    ) -> list[HygieneViolation]:
        """Rule 4: Ensure the look-elsewhere effect is reported."""
        violations: list[HygieneViolation] = []

        if search_space_size is not None and search_space_size > 1 and not reported:
            violations.append(
                HygieneViolation(
                    violation_type=ViolationType.LOOK_ELSEWHERE_OMITTED,
                    description=(
                        f"Searched {search_space_size} formulas but did not "
                        f"report the look-elsewhere effect."
                    ),
                    severity=2,
                    penalty_points=2,
                )
            )

        return violations

    def check_precision(
        self,
        *,
        claimed_precision_pct: float,
        actual_precision_pct: float,
        tolerance_ratio: float = 2.0,
    ) -> list[HygieneViolation]:
        """Detect precision inflation."""
        violations: list[HygieneViolation] = []

        if actual_precision_pct > claimed_precision_pct * tolerance_ratio:
            violations.append(
                HygieneViolation(
                    violation_type=ViolationType.PRECISION_INFLATION,
                    description=(
                        f"Claimed precision {claimed_precision_pct}% but "
                        f"actual deviation is {actual_precision_pct}%."
                    ),
                    severity=3,
                    penalty_points=3,
                )
            )

        return violations

    def check_silent_calibration(
        self,
        *,
        experimental_data_used: list[str] | None = None,
        disclosed_inputs: list[str] | None = None,
    ) -> list[HygieneViolation]:
        """Detect undisclosed use of experimental data."""
        violations: list[HygieneViolation] = []
        experimental_data_used = experimental_data_used or []
        disclosed_inputs = disclosed_inputs or []

        undisclosed = set(experimental_data_used) - set(disclosed_inputs)
        if undisclosed:
            violations.append(
                HygieneViolation(
                    violation_type=ViolationType.SILENT_CALIBRATION,
                    description=(
                        f"Experimental data used but not disclosed: "
                        f"{', '.join(sorted(undisclosed))}."
                    ),
                    severity=3,
                    penalty_points=3,
                )
            )

        return violations

    def full_audit(
        self,
        *,
        claimed_classification: str = "DERIVATION",
        fitted_parameters: list[str] | None = None,
        experimental_inputs: list[str] | None = None,
        constants_in: set[str] | None = None,
        constants_out: set[str] | None = None,
        is_algebraic_identity: bool = False,
        search_space_size: int | None = None,
        look_elsewhere_reported: bool = False,
        claimed_precision_pct: float | None = None,
        actual_precision_pct: float | None = None,
        experimental_data_used: list[str] | None = None,
        disclosed_inputs: list[str] | None = None,
    ) -> HygieneReport:
        """Run the complete epistemic hygiene audit on a claim."""
        report = HygieneReport()

        # Rule 1 & 2: derivation vs fit / prediction vs postdiction
        for v in self.check_derivation_vs_fit(
            claimed_classification=claimed_classification,
            fitted_parameters=fitted_parameters,
            experimental_inputs=experimental_inputs,
        ):
            report.add_violation(v)

        # Rule 3: tautology
        for v in self.check_tautology(
            constants_in=constants_in,
            constants_out=constants_out,
            is_algebraic_identity=is_algebraic_identity,
        ):
            report.add_violation(v)

        # Rule 4: look-elsewhere
        for v in self.check_look_elsewhere(
            search_space_size=search_space_size,
            reported=look_elsewhere_reported,
        ):
            report.add_violation(v)

        # Precision inflation
        if claimed_precision_pct is not None and actual_precision_pct is not None:
            for v in self.check_precision(
                claimed_precision_pct=claimed_precision_pct,
                actual_precision_pct=actual_precision_pct,
            ):
                report.add_violation(v)

        # Silent calibration
        for v in self.check_silent_calibration(
            experimental_data_used=experimental_data_used,
            disclosed_inputs=disclosed_inputs,
        ):
            report.add_violation(v)

        # Track calibrations
        for param in (fitted_parameters or []):
            report.calibrations.append(CalibrationRecord(parameter_name=param))

        # Track search spaces
        if search_space_size is not None:
            report.search_spaces.append(
                SearchSpaceRecord(total_formulas_searched=search_space_size)
            )

        return report
