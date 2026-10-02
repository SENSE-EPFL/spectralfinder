"""Temperature sensitivity of a selected window (reported, never used to
filter or rank).

For each target: alpha AT its selected peak (position held fixed) is recomputed at
T, T + dT, T + 2 dT, ... (full HAPI Voigt, same code path as stage 2), at
CONSTANT PRESSURE -- as in a pressure-regulated cell/cavity, so the number
density n = P/(kT) drops with T too (about -0.34 %/K near 296 K, the same
for every species). Everything else is line physics: S(T) (lower-state
population, partition function), Doppler and collisional widths.

Headline coefficient = (alpha(T + dT) - alpha(T)) / alpha(T) / dT, in %/K.
The full sweep is kept so one can see whether it stays linear.

For two targets of the same molecule (isotopologues, e.g. CH4 and (13C)H4),
the coefficient of their alpha RATIO is also given -- for an isotope-ratio
measurement that's the number that matters: a common T effect cancels, a
difference in lower-state energy does not.
"""
from __future__ import annotations

import numpy as np

from .lines import ensure_lines
from .spectrum import absorption_coefficient
from .species import molecule_id

_HAPI_INTENSITY_FLOOR = 1e-30


def temperature_dependence(window, targets: list, config) -> tuple:
    """Returns ({label: %/K}, {label: [(T, alpha), ...]}, {'a/b': %/K for isotopologue pairs})."""
    dT = config.temperature_step_K
    # At least T and T + dT: the headline coefficient needs both.
    temps = [config.T + k * dT for k in range(max(1, config.temperature_steps) + 1)]

    sweep, coeff = {}, {}
    for t in targets:
        pos = window.peak_position[t.label]
        table = ensure_lines(t.name, config.sigma_min, config.sigma_max)
        alphas = [
            float(absorption_coefficient(table, molecule_id(t.name), t.iso_id, t.mole_fraction,
                                         np.array([pos]), T, config.P, _HAPI_INTENSITY_FLOOR,
                                         t.abundance)[0])
            for T in temps
        ]
        sweep[t.label] = list(zip(temps, alphas))
        coeff[t.label] = 100 * (alphas[1] - alphas[0]) / alphas[0] / dT if alphas[0] > 0 else float("nan")

    ratio_coeff = {}
    for i, a in enumerate(targets):
        for b in targets[i + 1:]:
            if a.name != b.name:
                continue
            ra = [x for _, x in sweep[a.label]]
            rb = [x for _, x in sweep[b.label]]
            if min(ra[0], ra[1], rb[0], rb[1]) <= 0:
                continue
            r0, r1 = ra[0] / rb[0], ra[1] / rb[1]
            ratio_coeff[f"{a.label}/{b.label}"] = 100 * (r1 - r0) / r0 / dT
    return coeff, sweep, ratio_coeff
