"""Line-width physics.

All widths returned here are FULL widths at half maximum (FWHM), in cm^-1.
HITRAN's gamma_air / gamma_self columns are HALF widths (HWHM) at
T_ref=296 K, p_ref=1 atm -- doubling happens explicitly below, don't skip it.

Correction vs. the initial sketch: Doppler broadening depends on temperature
and molecular mass ONLY, never on pressure ("gamma_dop * P_atm" is not a
thing). Pressure/collisional broadening is the air+self term. The two
mechanisms are independent and combine as a Voigt profile, not a plain sum.
"""
from __future__ import annotations

from math import log, sqrt

from hapi import EnvironmentDependency_Gamma0, cBolts, cc, molecularMass

from .constants import P_REF, T_REF

_AMU_TO_G = 1.66053873e-24  # 1 amu in grams (CGS, consistent with hapi.cBolts/cc)

def doppler_fwhm(nu0: float, molec_id: int, iso_id: int, T: float) -> float:
    """Gaussian Doppler FWHM (cm^-1) at temperature T (K). Pressure-independent."""
    m_grams = molecularMass(molec_id, iso_id) * _AMU_TO_G
    return nu0 * sqrt(8 * log(2) * cBolts * T / (m_grams * cc**2))


def lorentz_fwhm(gamma_air: float, gamma_self: float, n_air: float,
                  T: float, P: float, self_fraction: float) -> float:
    """Collisional (pressure) FWHM (cm^-1).

    gamma_air, gamma_self: HITRAN reference HWHM (cm^-1/atm at 296 K, 1 atm).
    self_fraction: the absorber's own mole fraction (its partial pressure is
    self_fraction * P, the rest of P is treated as air).
    n_air is applied to both terms -- HITRAN .par files don't carry a
    separate n_self, this is the standard approximation.
    """
    gamma0_ref = (1 - self_fraction) * gamma_air + self_fraction * gamma_self
    hwhm = EnvironmentDependency_Gamma0(gamma0_ref, T, T_REF, P, P_REF, n_air)
    return 2 * hwhm


def voigt_fwhm(fwhm_lorentz: float, fwhm_gauss: float) -> float:
    """Olivero & Longbothum (1977) pseudo-Voigt FWHM approximation (~0.02% accurate)."""
    return 0.5346 * fwhm_lorentz + sqrt(0.2166 * fwhm_lorentz**2 + fwhm_gauss**2)
