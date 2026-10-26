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


def alpha_vs_temperature(target, pos: float, config, steps: int | None = None) -> list:
    """[(T, alpha at `pos`), ...] for T = T, T + dT, ..., T + steps*dT (default
    config.temperature_steps, at least 1: the headline coefficient needs T + dT)."""
    dT = config.temperature_step_K
    steps = config.temperature_steps if steps is None else steps
    temps = [config.T + k * dT for k in range(max(1, steps) + 1)]
    table = ensure_lines(target.name, config.sigma_min, config.sigma_max)
    return [
        (T, float(absorption_coefficient(table, molecule_id(target.name), target.iso_id, target.mole_fraction,
                                         np.array([pos]), T, config.P, _HAPI_INTENSITY_FLOOR,
                                         target.abundance)[0]))
        for T in temps
    ]


def coefficient(sweep: list) -> float:
    """Headline coefficient of a sweep, %/K (nan if alpha(T) is 0)."""
    (T0, a0), (T1, a1) = sweep[0], sweep[1]
    return 100 * (a1 - a0) / a0 / (T1 - T0) if a0 > 0 else float("nan")


def temperature_dependence(window, targets: list, config) -> tuple:
    """Returns ({label: %/K}, {label: [(T, alpha), ...]}, {'a/b': %/K for isotopologue pairs})."""
    dT = config.temperature_step_K
    sweep, coeff = {}, {}
    for t in targets:
        sweep[t.label] = alpha_vs_temperature(t, window.peak_position[t.label], config)
        coeff[t.label] = coefficient(sweep[t.label])

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
