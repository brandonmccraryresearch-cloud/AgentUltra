"""Tests for the core reasoning pipeline."""


from agent_ultra.core.pipeline import (
    Assumption,
    ClaimType,
    ComprehensionResult,
    ConfidenceGrade,
    GroundingResult,
    PipelineResult,
    ReasoningPipeline,
    ResultClassification,
    VerificationResult,
    VerificationStatus,
)


class TestComprehensionResult:
    def test_defaults(self):
        r = ComprehensionResult(statement="E = mc²")
        assert r.statement == "E = mc²"
        assert r.claim_type == ClaimType.CONJECTURE
        assert r.assumptions == []
        assert r.dependencies == []

    def test_with_assumptions(self):
        r = ComprehensionResult(
            statement="test",
            assumptions=[Assumption(description="flat spacetime")],
        )
        assert len(r.assumptions) == 1
        assert r.assumptions[0].explicit is True


class TestVerificationResult:
    def test_defaults(self):
        v = VerificationResult()
        assert v.status == VerificationStatus.UNRESOLVED
        assert v.formal_proof is None

    def test_verified(self):
        v = VerificationResult(
            status=VerificationStatus.VERIFIED,
            dimensional_consistent=True,
            limiting_cases_recovered=True,
        )
        assert v.status == VerificationStatus.VERIFIED


class TestReasoningPipeline:
    def test_run_produces_all_phases(self):
        pipeline = ReasoningPipeline()
        result = pipeline.run("The Schwarzschild metric solves the vacuum Einstein equations")
        assert isinstance(result, PipelineResult)
        assert result.comprehension is not None
        assert result.verification is not None
        assert result.grounding is not None
        assert result.synthesis is not None

    def test_quality_gates(self):
        pipeline = ReasoningPipeline()
        result = pipeline.run("test claim")
        gates = result.passed_quality_gates
        assert "dimensional_consistency" in gates
        assert "classification_assigned" in gates

    def test_auto_grade_high_score(self):
        v = VerificationResult(
            status=VerificationStatus.VERIFIED,
            dimensional_consistent=True,
            limiting_cases_recovered=True,
            alternative_derivations=["alternative_1"],
        )
        g = GroundingResult(
            classification=ResultClassification.DERIVATION,
            falsifiable=True,
            precision_percent=0.05,
        )
        grade = ReasoningPipeline._auto_grade(v, g)
        assert grade == ConfidenceGrade.A

    def test_auto_grade_low_score(self):
        v = VerificationResult()
        g = GroundingResult()
        grade = ReasoningPipeline._auto_grade(v, g)
        assert grade in (ConfidenceGrade.D, ConfidenceGrade.C, ConfidenceGrade.F)

    def test_custom_hooks(self):
        def comp_hook(result, **kw):
            result.claim_type = ClaimType.THEOREM
            return result

        pipeline = ReasoningPipeline(comprehension_hook=comp_hook)
        result = pipeline.run("test")
        assert result.comprehension.claim_type == ClaimType.THEOREM
