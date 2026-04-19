"""Tests for computation modules."""

import numpy as np
import pytest

from agent_ultra.computation.numerical import (
    bootstrap_error,
    eigenvalue_decomposition,
    fft_power_spectrum,
    jackknife_error,
    monte_carlo_integrate,
    richardson_extrapolation,
)
from agent_ultra.computation.symbolic import (
    brillouin_zone_integral,
    construct_root_system,
)


class TestRootSystems:
    def test_a2(self):
        rs = construct_root_system("A", 2)
        assert rs.algebra_type == "A"
        assert rs.rank == 2
        assert len(rs.simple_roots) == 2
        # A_2 (SU(3)) has 3 positive roots
        assert rs.num_positive_roots == 3

    def test_d4(self):
        rs = construct_root_system("D", 4)
        assert rs.algebra_type == "D"
        assert rs.rank == 4
        assert len(rs.simple_roots) == 4
        # D_4 has 12 positive roots
        assert rs.num_positive_roots == 12

    def test_g2(self):
        rs = construct_root_system("G", 2)
        assert rs.algebra_type == "G"
        assert rs.rank == 2
        assert rs.num_positive_roots == 6
        assert rs.dimension == 14

    def test_d_requires_rank_2(self):
        with pytest.raises(ValueError):
            construct_root_system("D", 1)


class TestBrillouinZone:
    def test_simple_integral(self):
        # Integrate f(k) = 1 over a cubic BZ — should give BZ volume
        lattice = np.eye(3)
        mean, err = brillouin_zone_integral(
            lambda k: np.ones(k.shape[0]),
            lattice,
            samples=50_000,
            seed=42,
        )
        bz_volume = (2 * np.pi) ** 3
        assert abs(mean - bz_volume) / bz_volume < 0.02


class TestMonteCarlo:
    def test_unit_cube(self):
        # Integral of f(x)=1 over [0,1]³ = 1
        result = monte_carlo_integrate(
            lambda x: np.ones(x.shape[0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            samples=100_000,
            seed=0,
        )
        assert abs(result.estimate - 1.0) < 0.05

    def test_known_integral(self):
        # Integral of x² over [0,1] = 1/3
        result = monte_carlo_integrate(
            lambda x: x[:, 0] ** 2,
            np.array([0.0]),
            np.array([1.0]),
            samples=500_000,
            seed=1,
        )
        assert abs(result.estimate - 1 / 3) < 0.01


class TestSpectral:
    def test_fft_sine(self):
        t = np.linspace(0, 1, 1024, endpoint=False)
        signal = np.sin(2 * np.pi * 10 * t)
        freqs, power = fft_power_spectrum(signal, dt=1 / 1024)
        peak_freq = freqs[np.argmax(power)]
        assert abs(peak_freq - 10.0) < 2.0

    def test_eigenvalue_hermitian(self):
        m = np.array([[2.0, 1.0], [1.0, 3.0]])
        vals, vecs = eigenvalue_decomposition(m, hermitian=True)
        assert len(vals) == 2
        # Eigenvalues of [[2,1],[1,3]] are (5±√5)/2
        assert abs(sorted(vals)[0] - (5 - 5**0.5) / 2) < 1e-10


class TestConvergence:
    def test_richardson(self):
        # f(h) = exact + c*h² with exact=1.0
        exact = 1.0
        hs = [0.5, 0.25, 0.125]
        vals = [exact + 2.0 * h**2 for h in hs]
        result = richardson_extrapolation(vals, hs)
        assert abs(result.extrapolated_value - exact) < 0.01


class TestStatistics:
    def test_bootstrap(self):
        data = np.random.default_rng(42).normal(5.0, 1.0, size=200)
        est, err = bootstrap_error(data, seed=42)
        assert abs(est - 5.0) < 0.3
        assert err > 0

    def test_jackknife(self):
        data = np.random.default_rng(42).normal(5.0, 1.0, size=200)
        est, err = jackknife_error(data)
        assert abs(est - 5.0) < 0.3
        assert err > 0
