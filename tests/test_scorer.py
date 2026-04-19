"""Tests for the evaluation scoring system."""

import pytest

from agent_ultra.evaluation.scorer import (
    ANTI_PATTERNS,
    RubricGrade,
    Scorer,
)


@pytest.fixture
def scorer():
    return Scorer()


class TestCriterionGrading:
    def test_derivation_from_axioms(self, scorer):
        s = scorer.grade_derivation(from_axioms=True)
        assert s.grade == RubricGrade.A
        assert s.points == 3

    def test_derivation_sketch(self, scorer):
        s = scorer.grade_derivation(sketch_with_gaps=True)
        assert s.grade == RubricGrade.B

    def test_verification_full(self, scorer):
        s = scorer.grade_verification(formal_proof=True, numerical_check=True)
        assert s.grade == RubricGrade.A

    def test_verification_numerical_only(self, scorer):
        s = scorer.grade_verification(numerical_check=True)
        assert s.grade == RubricGrade.B

    def test_precision_excellent(self, scorer):
        s = scorer.grade_precision(deviation_percent=0.05)
        assert s.grade == RubricGrade.A

    def test_precision_poor(self, scorer):
        s = scorer.grade_precision(deviation_percent=15.0)
        assert s.grade == RubricGrade.F

    def test_falsifiability_quantitative(self, scorer):
        s = scorer.grade_falsifiability(precise_quantitative=True)
        assert s.grade == RubricGrade.A

    def test_honesty_all_caveats(self, scorer):
        s = scorer.grade_honesty(all_caveats_stated=True)
        assert s.grade == RubricGrade.A


class TestScoreCard:
    def test_perfect_score(self, scorer):
        card = scorer.score(
            "claim_1",
            derivation_kwargs={"from_axioms": True},
            verification_kwargs={"formal_proof": True, "numerical_check": True},
            precision_kwargs={"deviation_percent": 0.01},
            falsifiability_kwargs={"precise_quantitative": True},
            honesty_kwargs={"all_caveats_stated": True},
        )
        assert card.raw_score == 15
        assert card.final_score == 15
        assert card.overall_grade == RubricGrade.A

    def test_with_penalties(self, scorer):
        card = scorer.score(
            "claim_2",
            derivation_kwargs={"from_axioms": True},
            verification_kwargs={"formal_proof": True, "numerical_check": True},
            precision_kwargs={"deviation_percent": 0.01},
            falsifiability_kwargs={"precise_quantitative": True},
            honesty_kwargs={"all_caveats_stated": True},
            anti_patterns=["overclaiming", "tautology_smuggling"],
        )
        assert card.total_penalty == 6
        assert card.final_score == 15 - 6

    def test_worst_score(self, scorer):
        card = scorer.score("claim_3")
        assert card.raw_score == 0
        assert card.overall_grade == RubricGrade.F

    def test_anti_pattern_points(self):
        assert ANTI_PATTERNS["overclaiming"] == 3
        assert ANTI_PATTERNS["look_elsewhere_blindness"] == 2
