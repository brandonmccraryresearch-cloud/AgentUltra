"""Tests for the knowledge base (constants, particles, cosmology)."""

from agent_ultra.knowledge.constants import (
    CODATA,
    get_constant,
    get_cosmological_parameter,
    get_particle,
)


class TestCODATA:
    def test_speed_of_light(self):
        c = get_constant("c")
        assert c is not None
        assert c.value == 299_792_458.0
        assert c.unit == "m/s"

    def test_fine_structure(self):
        alpha = get_constant("alpha")
        assert alpha is not None
        assert 7.29e-3 < alpha.value < 7.30e-3

    def test_unknown_constant(self):
        assert get_constant("nonexistent") is None

    def test_all_constants_have_units(self):
        for key, const in CODATA.items():
            assert const.unit, f"{key} is missing units"


class TestPDG:
    def test_electron(self):
        e = get_particle("electron")
        assert e is not None
        assert abs(e.mass_gev - 0.000_510_999) < 1e-6
        assert e.charge == -1.0

    def test_higgs(self):
        h = get_particle("higgs")
        assert h is not None
        assert 125.0 < h.mass_gev < 126.0

    def test_unknown_particle(self):
        assert get_particle("graviton") is None


class TestPlanck:
    def test_hubble(self):
        h0 = get_cosmological_parameter("H0")
        assert h0 is not None
        assert 60 < h0.value < 75

    def test_dark_energy(self):
        ol = get_cosmological_parameter("Omega_Lambda")
        assert ol is not None
        assert 0.6 < ol.value < 0.8
