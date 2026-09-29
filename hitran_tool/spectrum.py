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

from .lines import SLICE_MARGIN, sliced_table


def number_density(P: float, T: float) -> float:
    """Total number density n_total(P,T) (molec/cm^3) from the ideal gas law,
    via HAPI's own volumeConcentration -- reused directly rather than
    reimplemented, since absorption_coefficient() below relies on this exact
    same quantity internally (through HITRAN_units=False)."""
    return _volume_concentration(P, T)


def window_grid(lo: float, hi: float, step: float, extra_points=()) -> np.ndarray:
    """Regular grid over [lo, hi] plus `extra_points` (e.g. the exact line
    centers being evaluated), sorted and deduplicated.

    Why the extra points: at low pressure a line's FWHM (~0.008-0.02 cm^-1 at
    30 mbar) is comparable to any practical grid step, so a regular grid alone
    can miss a line's top by half a step and underestimate its peak by tens
    of percent. Adding the line centers themselves makes alpha AT the line
    exact, whatever the step. (A pressure-shifted center can sit a hair
    outside [lo, hi]; it's kept anyway.)
    """
    grid = np.arange(lo, hi + step / 2, step)
    return np.unique(np.concatenate([grid, np.asarray(list(extra_points), dtype=float)]))


def absorption_coefficient(table: str, molec_id: int, iso_id: int, mole_fraction: float,
                            grid: np.ndarray, T: float, P: float, s_cutoff: float,
                            abundance: float | None = None):
    """Return alpha (cm^-1) of ONE isotopologue on `grid` (cm^-1).

    mole_fraction: of the whole molecule (all isotopologues) -- it sets both
    the number density and the self-broadening share. The isotopologue's own
    share comes from HITRAN's natural abundance (built into its line
    intensities), or from `abundance` if given (HAPI rescales sw by
    abundance / natural_abundance).

    `table` is sliced to the grid's range (+ margin) and to this isotopologue
    first -- see lines.sliced_table for why: HAPI loops over every row of the
    table it's handed, in pure Python.
    """
    grid = np.asarray(grid, dtype=float)
    table = sliced_table(table, float(grid[0]), float(grid[-1]), iso_id=iso_id)
    # WavenumberWing: HAPI otherwise cuts every line at 50 half-widths from its center
    # (~0.3 cm^-1 at 30 mbar). For a saturated neighbour (e.g. an HDO line at 30% H2O,
    # alpha ~1e-5) the dropped wing is still ~1e-9 cm^-1 -- as big as the trace-gas
    # signals -- and shows as steps in the plotted curves. Summing every line of the
    # slice over the whole grid (wing = slice margin) keeps it consistent.
    component = (molec_id, iso_id) if abundance is None else (molec_id, iso_id, abundance)
    with contextlib.redirect_stdout(io.StringIO()):  # HAPI prints Diluent + timing on every call
        _, xsect_times_ntotal = absorptionCoefficient_Voigt(
            Components=[component],
            SourceTables=table,
            Environment={"T": T, "p": P},
            Diluent={"air": 1 - mole_fraction, "self": mole_fraction},
            WavenumberRange=[float(grid[0]), float(grid[-1])],
            WavenumberGrid=grid,
            WavenumberWing=SLICE_MARGIN,
            IntensityThreshold=s_cutoff,
            HITRAN_units=False,
        )
    return np.asarray(xsect_times_ntotal) * mole_fraction


def optical_depth(alpha_peak: float, L: float) -> float:
    """Optical depth tau = -ln(I/I0) = alpha_peak * L (Beer-Lambert), for a
    KNOWN, fixed path length L (cm) -- the calibration-instrument case:
    L is an input (your actual cell/cavity length), not something to solve for."""
    return alpha_peak * L
