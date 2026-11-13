"""Physics constants and experimental data interface.

Provides programmatic access to fundamental physical constants (CODATA),
particle data (PDG), and cosmological parameters (Planck). Values are
embedded for offline use; the agent may also query live MCP servers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhysicalConstant:
    """A fundamental physical constant with uncertainty."""

    name: str
    symbol: str
    value: float
    uncertainty: float
    unit: str
    source: str = "CODATA 2018"


@dataclass
class ParticleData:
    """Properties of a fundamental particle."""

    name: str
    mass_gev: float | None = None
    mass_uncertainty_gev: float | None = None
    lifetime_s: float | None = None
    charge: float = 0.0
    spin: float = 0.0
    source: str = "PDG 2024"


# ---------------------------------------------------------------------------
# CODATA 2018 fundamental constants
# ---------------------------------------------------------------------------

CODATA: dict[str, PhysicalConstant] = {
    "c": PhysicalConstant(
        name="speed of light in vacuum",
        symbol="c",
        value=299_792_458.0,
        uncertainty=0.0,
        unit="m/s",
    ),
    "hbar": PhysicalConstant(
        name="reduced Planck constant",
        symbol="ℏ",
        value=1.054_571_817e-34,
        uncertainty=0.0,
        unit="J·s",
    ),
    "G": PhysicalConstant(
        name="Newtonian constant of gravitation",
        symbol="G",
        value=6.674_30e-11,
        uncertainty=1.5e-15,
        unit="m³/(kg·s²)",
    ),
    "e": PhysicalConstant(
        name="elementary charge",
        symbol="e",
        value=1.602_176_634e-19,
        uncertainty=0.0,
        unit="C",
    ),
    "k_B": PhysicalConstant(
        name="Boltzmann constant",
        symbol="k_B",
        value=1.380_649e-23,
        uncertainty=0.0,
        unit="J/K",
    ),
    "alpha": PhysicalConstant(
        name="fine-structure constant",
        symbol="α",
        value=7.297_352_5693e-3,
        uncertainty=1.1e-12,
        unit="dimensionless",
    ),
    "alpha_inv": PhysicalConstant(
        name="inverse fine-structure constant",
        symbol="α⁻¹",
        value=137.035_999_084,
        uncertainty=0.000_000_021,
        unit="dimensionless",
    ),
    "m_e": PhysicalConstant(
        name="electron mass",
        symbol="m_e",
        value=9.109_383_7015e-31,
        uncertainty=2.8e-40,
        unit="kg",
    ),
    "m_p": PhysicalConstant(
        name="proton mass",
        symbol="m_p",
        value=1.672_621_923_69e-27,
        uncertainty=5.1e-37,
        unit="kg",
    ),
    "N_A": PhysicalConstant(
        name="Avogadro constant",
        symbol="N_A",
        value=6.022_140_76e23,
        uncertainty=0.0,
        unit="1/mol",
    ),
    "m_Pl": PhysicalConstant(
        name="Planck mass",
        symbol="m_Pl",
        value=2.176_434e-8,
        uncertainty=2.4e-13,
        unit="kg",
    ),
    "l_Pl": PhysicalConstant(
        name="Planck length",
        symbol="l_Pl",
        value=1.616_255e-35,
        uncertainty=1.8e-40,
        unit="m",
    ),
    "t_Pl": PhysicalConstant(
        name="Planck time",
        symbol="t_Pl",
        value=5.391_247e-44,
        uncertainty=6.0e-49,
        unit="s",
    ),
}


# ---------------------------------------------------------------------------
# PDG particle data (selected entries)
# ---------------------------------------------------------------------------

PDG_PARTICLES: dict[str, ParticleData] = {
    "electron": ParticleData(
        name="electron", mass_gev=0.000_510_999, charge=-1.0, spin=0.5
    ),
    "muon": ParticleData(
        name="muon",
        mass_gev=0.105_658,
        lifetime_s=2.196_98e-6,
        charge=-1.0,
        spin=0.5,
    ),
    "tau": ParticleData(
        name="tau",
        mass_gev=1.776_86,
        lifetime_s=2.903e-13,
        charge=-1.0,
        spin=0.5,
    ),
    "proton": ParticleData(
        name="proton", mass_gev=0.938_272, charge=1.0, spin=0.5
    ),
    "neutron": ParticleData(
        name="neutron",
        mass_gev=0.939_565,
        lifetime_s=878.4,
        charge=0.0,
        spin=0.5,
    ),
    "W_boson": ParticleData(
        name="W boson", mass_gev=80.377, charge=1.0, spin=1.0
    ),
    "Z_boson": ParticleData(
        name="Z boson", mass_gev=91.1876, charge=0.0, spin=1.0
    ),
    "higgs": ParticleData(
        name="Higgs boson", mass_gev=125.25, charge=0.0, spin=0.0
    ),
    "top_quark": ParticleData(
        name="top quark", mass_gev=172.69, charge=2.0 / 3.0, spin=0.5
    ),
    "bottom_quark": ParticleData(
        name="bottom quark", mass_gev=4.18, charge=-1.0 / 3.0, spin=0.5
    ),
}


# ---------------------------------------------------------------------------
# Planck 2018 cosmological parameters
# ---------------------------------------------------------------------------

PLANCK_COSMOLOGY: dict[str, PhysicalConstant] = {
    "H0": PhysicalConstant(
        name="Hubble constant",
        symbol="H₀",
        value=67.36,
        uncertainty=0.54,
        unit="km/s/Mpc",
        source="Planck 2018",
    ),
    "Omega_b": PhysicalConstant(
        name="baryon density parameter",
        symbol="Ω_b h²",
        value=0.02237,
        uncertainty=0.00015,
        unit="dimensionless",
        source="Planck 2018",
    ),
    "Omega_c": PhysicalConstant(
        name="cold dark matter density parameter",
        symbol="Ω_c h²",
        value=0.1200,
        uncertainty=0.0012,
        unit="dimensionless",
        source="Planck 2018",
    ),
    "Omega_Lambda": PhysicalConstant(
        name="dark energy density parameter",
        symbol="Ω_Λ",
        value=0.6847,
        uncertainty=0.0073,
        unit="dimensionless",
        source="Planck 2018",
    ),
    "n_s": PhysicalConstant(
        name="scalar spectral index",
        symbol="n_s",
        value=0.9649,
        uncertainty=0.0042,
        unit="dimensionless",
        source="Planck 2018",
    ),
    "sigma_8": PhysicalConstant(
        name="matter fluctuation amplitude",
        symbol="σ₈",
        value=0.8111,
        uncertainty=0.0060,
        unit="dimensionless",
        source="Planck 2018",
    ),
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_constant(name: str) -> PhysicalConstant | None:
    """Look up a CODATA fundamental constant by key."""
    return CODATA.get(name)


def get_particle(name: str) -> ParticleData | None:
    """Look up PDG particle data by name."""
    return PDG_PARTICLES.get(name)


def get_cosmological_parameter(name: str) -> PhysicalConstant | None:
    """Look up a Planck 2018 cosmological parameter by key."""
    return PLANCK_COSMOLOGY.get(name)
