"""Numerical computation engine.

Provides helpers for Monte Carlo integration, spectral methods,
convergence analysis, and statistical error estimation used by the
research agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monte Carlo integration
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """Result of a Monte Carlo integration."""

    estimate: float
    std_error: float
    samples: int
    variance: float = 0.0


def monte_carlo_integrate(
    integrand: Callable[[np.ndarray], np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    samples: int = 1_000_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Monte Carlo integration over a hyper-rectangular domain.

    Parameters
    ----------
    integrand:
        f(x) where x has shape (N, d) → returns shape (N,).
    lower, upper:
        Bounds of the integration domain (shape (d,)).
    samples:
        Number of random samples.
    seed:
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    d = len(lower)
    volume = float(np.prod(upper - lower))

    points = rng.uniform(lower, upper, size=(samples, d))
    values = np.asarray(integrand(points), dtype=np.float64)

    mean = float(np.mean(values)) * volume
    var = float(np.var(values, ddof=1))
    std_error = float(np.sqrt(var / samples)) * volume

    return MonteCarloResult(
        estimate=mean, std_error=std_error, samples=samples, variance=var
    )


# ---------------------------------------------------------------------------
# Spectral methods
# ---------------------------------------------------------------------------

def fft_power_spectrum(signal: np.ndarray, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute the one-sided power spectrum of a real signal.

    Returns (frequencies, power) arrays.
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    spectrum = np.fft.rfft(signal)
    power = np.abs(spectrum) ** 2 / n
    return freqs, power


def eigenvalue_decomposition(
    matrix: np.ndarray, *, hermitian: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors.

    Uses the Hermitian-specific routine when ``hermitian=True`` for
    better numerical stability.
    """
    if hermitian:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceResult:
    """Result of a Richardson extrapolation."""

    extrapolated_value: float
    estimated_order: float
    values: list[float] = field(default_factory=list)
    step_sizes: list[float] = field(default_factory=list)


def richardson_extrapolation(
    values: list[float],
    step_sizes: list[float],
    *,
    expected_order: float | None = None,
) -> ConvergenceResult:
    """Perform Richardson extrapolation to estimate the continuum limit.

    Requires at least two values at different step sizes.  If three or
    more are provided the convergence order is estimated.
    """
    if len(values) < 2:
        raise ValueError("Need at least two values for Richardson extrapolation")
    if len(step_sizes) != len(values):
        raise ValueError(
            f"step_sizes and values must have the same length "
            f"(got {len(step_sizes)} step sizes and {len(values)} values)"
        )
    if len(step_sizes) < 2:
        raise ValueError("Need at least two step sizes for Richardson extrapolation")
    if any(h <= 0 for h in step_sizes):
        raise ValueError("All step sizes must be positive")

    # Estimate order from three finest values if available
    if len(values) >= 3 and expected_order is None:
        _h0, h1, h2 = step_sizes[-3], step_sizes[-2], step_sizes[-1]
        f0, f1, f2 = values[-3], values[-2], values[-1]
        denom = np.log(h1 / h2)
        if abs(denom) > 1e-15 and abs(f1 - f2) > 1e-30 and abs(f0 - f1) > 1e-30:
            order = float(np.log(abs((f0 - f1) / (f1 - f2))) / denom)
        else:
            order = 2.0  # default assumption
    else:
        order = expected_order if expected_order is not None else 2.0

    # Extrapolate from the two finest values. The denominator (1 - ratio)
    # vanishes when the two finest step sizes are equal, or when the
    # convergence order is zero (ratio == 1) — both mean no extrapolation
    # is possible, so reject them instead of dividing by zero.
    h_fine, h_coarse = step_sizes[-1], step_sizes[-2]
    f_fine, f_coarse = values[-1], values[-2]
    if h_fine == h_coarse:
        raise ValueError(
            "The two finest step sizes must differ for Richardson extrapolation"
        )
    ratio = (h_fine / h_coarse) ** order
    if abs(1.0 - ratio) < 1e-15:
        raise ValueError(
            "Richardson extrapolation is undefined here: the convergence "
            f"order ({order}) makes the extrapolation ratio equal to 1"
        )
    extrapolated = float((f_fine - ratio * f_coarse) / (1.0 - ratio))

    return ConvergenceResult(
        extrapolated_value=extrapolated,
        estimated_order=order,
        values=list(values),
        step_sizes=list(step_sizes),
    )


# ---------------------------------------------------------------------------
# Statistical error estimation
# ---------------------------------------------------------------------------

def bootstrap_error(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_resamples: int = 1000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Estimate the standard error of a statistic via bootstrapping.

    Returns (estimate, std_error).
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    estimates = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = data[rng.integers(0, n, size=n)]
        estimates[i] = statistic(sample)
    return float(np.mean(estimates)), float(np.std(estimates, ddof=1))


def jackknife_error(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> tuple[float, float]:
    """Estimate the standard error of a statistic via jackknife resampling.

    Returns (estimate, std_error).
    """
    n = len(data)
    full_stat = statistic(data)
    jack_stats = np.empty(n)
    for i in range(n):
        subset = np.concatenate([data[:i], data[i + 1 :]])
        jack_stats[i] = statistic(subset)
    bias = (n - 1) * (float(np.mean(jack_stats)) - full_stat)
    variance = ((n - 1) / n) * float(np.sum((jack_stats - np.mean(jack_stats)) ** 2))
    return full_stat - bias, float(np.sqrt(variance))
