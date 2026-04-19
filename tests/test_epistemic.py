"""Tests for the epistemic hygiene module."""

import pytest

from agent_ultra.core.epistemic import (
    EpistemicHygieneChecker,
    HygieneReport,
    ViolationType,
)


@pytest.fixture
def checker():
    return EpistemicHygieneChecker()


class TestDerivationVsFit:
    def test_fit_claimed_as_derivation(self, checker):
        violations = checker.check_derivation_vs_fit(
            claimed_classification="DERIVATION",
            fitted_parameters=["lambda_QCD"],
        )
        assert len(violations) == 1
        assert violations[0].violation_type == ViolationType.FIT_AS_DERIVATION
        assert violations[0].penalty_points == 3

    def test_genuine_derivation(self, checker):
        violations = checker.check_derivation_vs_fit(
            claimed_classification="DERIVATION",
            fitted_parameters=[],
        )
        assert len(violations) == 0

    def test_postdiction_as_prediction(self, checker):
        violations = checker.check_derivation_vs_fit(
            claimed_classification="PREDICTION",
            experimental_inputs=["alpha_em"],
        )
        assert len(violations) == 1
        assert violations[0].violation_type == ViolationType.POSTDICTION_AS_PREDICTION


class TestTautology:
    def test_passthrough_constants(self, checker):
        violations = checker.check_tautology(
            constants_in={"c", "hbar"},
            constants_out={"c", "hbar"},
        )
        assert len(violations) == 1
        assert violations[0].violation_type == ViolationType.TAUTOLOGY_AS_PREDICTION

    def test_no_passthrough(self, checker):
        violations = checker.check_tautology(
            constants_in={"c", "hbar"},
            constants_out={"alpha"},
        )
        assert len(violations) == 0

    def test_algebraic_identity(self, checker):
        violations = checker.check_tautology(is_algebraic_identity=True)
        assert any(
            v.violation_type == ViolationType.TAUTOLOGY_AS_PREDICTION
            for v in violations
        )


class TestLookElsewhere:
    def test_unreported(self, checker):
        violations = checker.check_look_elsewhere(
            search_space_size=10_000,
            reported=False,
        )
        assert len(violations) == 1
        assert violations[0].violation_type == ViolationType.LOOK_ELSEWHERE_OMITTED

    def test_reported(self, checker):
        violations = checker.check_look_elsewhere(
            search_space_size=10_000,
            reported=True,
        )
        assert len(violations) == 0


class TestPrecision:
    def test_precision_inflation(self, checker):
        violations = checker.check_precision(
            claimed_precision_pct=0.2,
            actual_precision_pct=11.0,
        )
        assert len(violations) == 1
        assert violations[0].violation_type == ViolationType.PRECISION_INFLATION

    def test_accurate_precision(self, checker):
        violations = checker.check_precision(
            claimed_precision_pct=1.0,
            actual_precision_pct=0.8,
        )
        assert len(violations) == 0


class TestSilentCalibration:
    def test_undisclosed(self, checker):
        violations = checker.check_silent_calibration(
            experimental_data_used=["m_Z", "alpha_s"],
            disclosed_inputs=["m_Z"],
        )
        assert len(violations) == 1
        assert violations[0].violation_type == ViolationType.SILENT_CALIBRATION


class TestFullAudit:
    def test_clean_audit(self, checker):
        report = checker.full_audit(
            claimed_classification="DERIVATION",
            fitted_parameters=[],
        )
        assert isinstance(report, HygieneReport)
        assert report.is_clean

    def test_dirty_audit(self, checker):
        report = checker.full_audit(
            claimed_classification="DERIVATION",
            fitted_parameters=["g_s"],
            search_space_size=5000,
            look_elsewhere_reported=False,
            claimed_precision_pct=0.1,
            actual_precision_pct=15.0,
        )
        assert not report.is_clean
        assert report.total_penalty > 0
        assert len(report.violations) >= 2
