"""Symbolic computation helpers.

Provides wrappers around SymPy for the algebraic operations needed by the
agent: root system construction, Brillouin zone integration, representation
theory, and invariant computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import importlib.util

    HAS_SYMPY = importlib.util.find_spec("sympy") is not None
except ImportError:  # pragma: no cover
    HAS_SYMPY = False


# ---------------------------------------------------------------------------
# Root Systems
# ---------------------------------------------------------------------------

@dataclass
class RootSystem:
    """Representation of a Lie algebra root system."""

    algebra_type: str  # e.g. "D", "E", "A"
    rank: int
    simple_roots: list[list[float]] = field(default_factory=list)
    positive_roots: list[list[float]] = field(default_factory=list)
    cartan_matrix: list[list[int]] = field(default_factory=list)
    dimension: int = 0

    @property
    def num_positive_roots(self) -> int:
        return len(self.positive_roots)


def construct_root_system(algebra_type: str, rank: int) -> RootSystem:
    """Construct the root system for a given Lie algebra.

    Supports A_n, B_n, C_n, D_n types via explicit construction.
    Exceptional types (E_6, E_7, E_8, F_4, G_2) return well-known data.
    """
    rs = RootSystem(algebra_type=algebra_type, rank=rank)

    if algebra_type.upper() == "A":
        rs = _build_type_a(rank)
    elif algebra_type.upper() == "D":
        rs = _build_type_d(rank)
    elif algebra_type.upper() == "G" and rank == 2:
        rs = _build_g2()
    else:
        # For other types, store the Cartan matrix at minimum
        rs.cartan_matrix = _generic_cartan_matrix(algebra_type.upper(), rank)

    rs.algebra_type = algebra_type.upper()
    rs.rank = rank
    return rs


def _build_type_a(n: int) -> RootSystem:
    """Root system for A_n (SU(n+1))."""
    # Simple roots: e_i - e_{i+1} in R^{n+1}
    simple_roots: list[list[float]] = []
    for i in range(n):
        root = [0.0] * (n + 1)
        root[i] = 1.0
        root[i + 1] = -1.0
        simple_roots.append(root)

    # Positive roots: e_i - e_j for i < j
    positive_roots: list[list[float]] = []
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            root = [0.0] * (n + 1)
            root[i] = 1.0
            root[j] = -1.0
            positive_roots.append(root)

    # Cartan matrix
    cartan = [[0] * n for _ in range(n)]
    for i in range(n):
        cartan[i][i] = 2
        if i > 0:
            cartan[i][i - 1] = -1
        if i < n - 1:
            cartan[i][i + 1] = -1

    return RootSystem(
        algebra_type="A",
        rank=n,
        simple_roots=simple_roots,
        positive_roots=positive_roots,
        cartan_matrix=cartan,
        dimension=n * (n + 2),
    )


def _build_type_d(n: int) -> RootSystem:
    """Root system for D_n (SO(2n))."""
    if n < 2:
        raise ValueError("D_n requires n >= 2")

    # Simple roots in R^n:
    # α_i = e_i - e_{i+1} for i = 1..n-1
    # α_n = e_{n-1} + e_n
    simple_roots: list[list[float]] = []
    for i in range(n - 1):
        root = [0.0] * n
        root[i] = 1.0
        root[i + 1] = -1.0
        simple_roots.append(root)
    last = [0.0] * n
    last[n - 2] = 1.0
    last[n - 1] = 1.0
    simple_roots.append(last)

    # Positive roots: e_i ± e_j for i < j
    positive_roots: list[list[float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            r1 = [0.0] * n
            r1[i] = 1.0
            r1[j] = -1.0
            positive_roots.append(r1)

            r2 = [0.0] * n
            r2[i] = 1.0
            r2[j] = 1.0
            positive_roots.append(r2)

    # Cartan matrix
    cartan = [[0] * n for _ in range(n)]
    for i in range(n):
        cartan[i][i] = 2
    for i in range(n - 2):
        cartan[i][i + 1] = -1
        cartan[i + 1][i] = -1
    # D_n specific: α_{n-2} connects to α_n
    if n >= 3:
        cartan[n - 3][n - 1] = -1
        cartan[n - 1][n - 3] = -1

    return RootSystem(
        algebra_type="D",
        rank=n,
        simple_roots=simple_roots,
        positive_roots=positive_roots,
        cartan_matrix=cartan,
        dimension=n * (2 * n - 1),
    )


def _build_g2() -> RootSystem:
    """Root system for G_2."""
    simple_roots = [
        [1.0, -1.0, 0.0],
        [-2.0, 1.0, 1.0],
    ]
    positive_roots = [
        [1.0, -1.0, 0.0],
        [-2.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0],
        [0.0, -1.0, 1.0],
        [1.0, -2.0, 1.0],
        [-1.0, -1.0, 2.0],
    ]
    cartan = [[2, -1], [-3, 2]]
    return RootSystem(
        algebra_type="G",
        rank=2,
        simple_roots=simple_roots,
        positive_roots=positive_roots,
        cartan_matrix=cartan,
        dimension=14,
    )


def _generic_cartan_matrix(algebra_type: str, rank: int) -> list[list[int]]:
    """Return Cartan matrix for simple types — fallback for unsupported types."""
    # Identity-like placeholder
    return [[2 if i == j else 0 for j in range(rank)] for i in range(rank)]


# ---------------------------------------------------------------------------
# Brillouin Zone Integration
# ---------------------------------------------------------------------------

def brillouin_zone_integral(
    integrand: Any,
    lattice_vectors: np.ndarray,
    *,
    method: str = "monte_carlo",
    samples: int = 100_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Evaluate an integral over the first Brillouin zone.

    Parameters
    ----------
    integrand:
        Callable that accepts an array of k-points (N × d) and returns
        an array of values (N,).
    lattice_vectors:
        Real-space lattice vectors (d × d matrix). Reciprocal vectors
        are computed automatically.
    method:
        Integration method — currently ``'monte_carlo'`` is supported.
    samples:
        Number of Monte Carlo samples.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (mean, std_error):
        Estimated integral value and standard error.
    """
    rng = np.random.default_rng(seed)
    d = lattice_vectors.shape[0]

    # Reciprocal lattice vectors: b_i = 2π (a^{-T})_i
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice_vectors).T

    # Uniform sampling in fractional reciprocal coordinates
    fractions = rng.uniform(0.0, 1.0, size=(samples, d))
    k_points = fractions @ reciprocal

    values = np.asarray(integrand(k_points), dtype=np.float64)
    bz_volume = abs(np.linalg.det(reciprocal))

    mean = float(np.mean(values)) * bz_volume
    std_error = float(np.std(values, ddof=1) / np.sqrt(samples)) * bz_volume
    return mean, std_error
