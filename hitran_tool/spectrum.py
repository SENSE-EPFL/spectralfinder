"""Absorption coefficient (Beer-Lambert alpha, cm^-1) and optical-depth helper.

alpha(sigma) = N * S(sigma)   [alpha = N * S * lineshape, integrated over the
Voigt line shapes of every line in range], with N the absorber's own number
density (molec/cm^3), NOT the total gas density.

HAPI's absorptionCoefficient_Voigt(..., HITRAN_units=False) multiplies the
cross section by volumeConcentration(p, T) = n_total(p, T) from the ideal gas
law -- i.e. it implicitly assumes the absorber is the ONLY gas present
(mole fraction = 1). For a trace species at mole fraction x, N = x *
n_total(p, T), so the array HAPI returns must still be scaled by x. This is
exactly the unit trap flagged in the original notes ("mic-mac d'unite"):
forgetting this factor overstates trace-gas absorption by 1/x (e.g. 1e5x for
a 10 ppm gas).
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
from hapi import absorptionCoefficient_Voigt
from hapi import volumeConcentration as _volume_concentration

from .lines import sliced_table


def number_density(P: float, T: float) -> float:
    """Total number density n_total(P,T) (molec/cm^3) from the ideal gas law,
    via HAPI's own volumeConcentration -- reused directly rather than
    reimplemented, since absorption_coefficient() below relies on this exact
    same quantity internally (through HITRAN_units=False)."""
    return _volume_concentration(P, T)


def absorption_coefficient(table: str, molec_id: int, iso_id: int, mole_fraction: float,
                            sigma_min: float, sigma_max: float, T: float, P: float,
                            step: float, s_cutoff: float):
    """Return (nu_grid, alpha[cm^-1]) for one species over [sigma_min, sigma_max].

    `table` is sliced to [sigma_min, sigma_max] (+ margin) first -- see
    lines.sliced_table for why: without it, a table whose cache has grown to
    cover a much wider range than what's asked here pays for every one of
    its rows on every call, not just the ones in range.
    """
    table = sliced_table(table, sigma_min, sigma_max)
    with contextlib.redirect_stdout(io.StringIO()):  # HAPI prints Diluent + timing on every call
        nu, xsect_times_ntotal = absorptionCoefficient_Voigt(
            Components=[(molec_id, iso_id)],
            SourceTables=table,
            Environment={"T": T, "p": P},
            Diluent={"air": 1 - mole_fraction, "self": mole_fraction},
            WavenumberRange=[sigma_min, sigma_max],
            WavenumberStep=step,
            IntensityThreshold=s_cutoff,
            HITRAN_units=False,
        )
    alpha = np.asarray(xsect_times_ntotal) * mole_fraction
    return np.asarray(nu), alpha


def optical_depth(alpha_peak: float, L: float) -> float:
    """Optical depth tau = -ln(I/I0) = alpha_peak * L (Beer-Lambert), for a
    KNOWN, fixed path length L (cm) -- the calibration-instrument case:
    L is an input (your actual cell/cavity length), not something to solve for."""
    return alpha_peak * L
