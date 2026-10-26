"""Background-interferer check for a selected window. The interference ratio
is reported only, never used to rank; the interferers' spectra DO feed the
saturation filter (total transmission across the window, see
find_combined_windows).

Who counts as an interferer:
  - every molecule of the background list (atm_standard.csv), at its
    standard mole fraction, ALL its isotopologues (each reported separately:
    '(13C)(16O)2' is not the same molecule spectrally as '(12C)(16O)2');
  - the OTHER isotopologues of each target molecule, at the target's own
    mole fraction (e.g. target H2O at 30% -> HD(16O), H2(18O)... at 30% H2O
    with natural abundance). An isotopologue that is itself a target isn't
    in this list -- it's passed separately (target_alpha) and counted in the
    ratio of the OTHER targets only.
A background molecule that is also a target is taken from the targets (its
concentration wins), so H2O isn't counted once at 30% and again at 1.17%.

What's measured, per window:
  - per interferer: its own peak alpha in the window (+ position), and its
    transmission exp(-peak_alpha * L_cm) at the instrument's path length;
  - per target: alpha of every OTHER species AT that target's selected
    peak -- that's what actually biases the measurement -- summed, divided
    by the target's own alpha there ('interference ratio'), plus which one
    contributes most. "Other species" includes the other TARGETS: the
    separation check only compares the selected peaks' positions, so e.g.
    the wing of a strong H2O line under a CH4 peak isn't caught anywhere else.
  - per background molecule: up to which mole fraction the window still
    passes interference_max at every target (background_limits) -- a warning
    for unusual environments, where one of them can be far above standard.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, inf

import numpy as np

from .lines import ensure_lines, isotopologues_present
from .spectrum import absorption_coefficient
from .species import isotopologue_ids, molecule_id, species_label

# Keeps numerically-zero lines out of HAPI's sum; not a physical cutoff.
_HAPI_INTENSITY_FLOOR = 1e-30


@dataclass
class InterfererResult:
    label: str  # 'CO2', '(13C)(16O)2', 'HD(16O)', ...
    molecule: str
    iso_id: int
    mole_fraction: float  # of the whole molecule
    peak_alpha: float  # max alpha in the window, cm^-1
    peak_position: float  # where that max is, cm^-1
    transmission: float  # exp(-peak_alpha * L_cm)
    alpha_at_target: dict  # {target_label: this interferer's alpha at that target's peak, cm^-1}


@dataclass
class BackgroundLimit:
    molecule: str
    mole_fraction: float  # standard, from the background list
    max_mole_fraction: float  # above it, `target`'s interference exceeds interference_max
    target: str  # the target it breaks first

    @property
    def factor(self) -> float:
        """How many times its standard mole fraction the molecule can rise (< 1: already too much)."""
        return self.max_mole_fraction / self.mole_fraction


def target_molecules(targets: list) -> dict:
    """{molecule: (mole_fraction, {iso_ids that are targets})}. Raises if two isotopologue
    targets of one molecule disagree on its mole fraction: it's the WHOLE molecule's, so it
    can't differ between them (use TargetSpecies.abundance for a non-natural composition)."""
    out = {}
    for t in targets:
        x, isos = out.setdefault(t.name, (t.mole_fraction, set()))
        if x != t.mole_fraction:
            raise ValueError(f"targets of {t.name} disagree on its mole fraction ({x} vs {t.mole_fraction})")
        isos.add(t.iso_id)
    return out


def interferer_list(targets: list, background: dict) -> list:
    """[(molecule, mole_fraction, excluded_iso_ids)], see module docstring."""
    molecules = target_molecules(targets)
    out = [(mol, x, set()) for mol, x in background.items() if mol not in molecules]
    out += [(mol, x, isos) for mol, (x, isos) in molecules.items()]
    return out


def ensure_interferer_lines(interferers: list, sigma_min: float, sigma_max: float) -> None:
    """Fetch every interferer's lines up front (one-time download, then cached)."""
    for mol, _, _ in interferers:
        ensure_lines(mol, sigma_min, sigma_max)


def _molecule_alphas(mol: str, x: float, excluded: set, grid: np.ndarray, config):
    """Yield (iso_id, alpha on grid) for every isotopologue of `mol` (but `excluded`)
    with lines near the grid and a non-zero absorption on it, at mole fraction x."""
    table = ensure_lines(mol, config.sigma_min, config.sigma_max)
    m_id = molecule_id(mol)
    known = set(isotopologue_ids(mol))
    for iso in isotopologues_present(table, float(grid[0]), float(grid[-1])):
        if iso in excluded or iso not in known:
            continue
        alpha = absorption_coefficient(table, m_id, iso, x, grid, config.T, config.P, _HAPI_INTENSITY_FLOOR)
        if alpha.size and float(alpha.max()) > 0:
            yield iso, alpha


def compute_interference(window, grid: np.ndarray, interferers: list, config,
                         target_alpha: dict) -> tuple:
    """target_alpha: {target_label: alpha on grid}, counted as interferers of the
    OTHER targets. Returns (list[InterfererResult] sorted strongest first,
                {label: alpha on grid} for each of them,
                {target_label: interference ratio},
                {target_label: label of the main interferer at that target, or None})."""

    def at_targets(alpha):  # value at each target's peak -- peaks sit BETWEEN grid points
        return {name: float(np.interp(pos, grid, alpha)) for name, pos in window.peak_position.items()}

    results, spectra = [], {}
    for mol, x, excluded in interferers:
        for iso, alpha in _molecule_alphas(mol, x, excluded, grid, config):
            k = int(np.argmax(alpha))
            label = species_label(mol, iso)
            results.append(InterfererResult(
                label=label, molecule=mol, iso_id=iso, mole_fraction=x,
                peak_alpha=float(alpha[k]), peak_position=float(grid[k]),
                transmission=exp(-float(alpha[k]) * config.L_cm),
                alpha_at_target=at_targets(alpha),
            ))
            spectra[label] = alpha

    results.sort(key=lambda r: r.peak_alpha, reverse=True)

    ratio, main = {}, {}
    for name in window.peak_position:
        contributions = [(r.alpha_at_target[name], r.label) for r in results]
        contributions += [(float(np.interp(window.peak_position[name], grid, alpha)), other)
                          for other, alpha in target_alpha.items() if other != name]
        total = sum(a for a, _ in contributions)
        ratio[name] = total / window.peak_alpha[name]
        main[name] = max(contributions)[1] if contributions and total > 0 else None
    return results, spectra, ratio, main


def background_limits(window, grid: np.ndarray, background: dict, config) -> list:
    """[BackgroundLimit] for every background molecule that would push some target's
    interference above config.interference_max before reaching a mole fraction of 1,
    most critical (lowest factor) first. `window` must have its interferers and
    interference computed; `background` must not hold target molecules (their
    isotopologues scale with the target, not independently).

    A trace gas's alpha is proportional to its mole fraction, so each molecule's
    contribution at a target -- summed over its isotopologues -- is scaled up until it
    fills the budget interference_max * alpha_target left by everything else: no new
    spectrum. Above _EXACT_ABOVE that's no longer true (its self-broadening reshapes the
    lines: CO2 at 50% came out ~15% off), so the limit is refined on its real spectrum
    at the estimated limit, a few times."""
    out = []
    for mol, x in background.items():
        at_target = {name: sum(r.alpha_at_target[name] for r in window.interferers if r.molecule == mol)
                     for name in window.peak_alpha}
        rest = {name: window.interference[name] * peak - at_target[name]
                for name, peak in window.peak_alpha.items()}
        limit, target = _lowest_limit(window, x, at_target, rest, config.interference_max)
        for _ in range(_REFINE_STEPS):
            if not _EXACT_ABOVE < limit < inf:
                break
            x_trial = min(limit, 1.0)
            at_trial = dict.fromkeys(window.peak_alpha, 0.0)
            for _, alpha in _molecule_alphas(mol, x_trial, set(), grid, config):
                for name, pos in window.peak_position.items():
                    at_trial[name] += float(np.interp(pos, grid, alpha))
            limit, target = _lowest_limit(window, x_trial, at_trial, rest, config.interference_max)
        if limit < 1:
            out.append(BackgroundLimit(mol, x, limit, target))
    out.sort(key=lambda b: b.factor)
    return out


_EXACT_ABOVE = 1e-2  # mole fraction above which a background limit is refined on real spectra
_REFINE_STEPS = 3


def _lowest_limit(window, x: float, at_target: dict, rest: dict, interference_max: float) -> tuple:
    """(mole fraction at which the first target reaches interference_max, that target),
    scaling linearly from `at_target` = the molecule's alpha at each target at mole
    fraction x; (inf, None) if it absorbs at none of them."""
    best = (inf, None)
    for name, peak in window.peak_alpha.items():
        if at_target[name] > 0:
            best = min(best, (x * max(interference_max * peak - rest[name], 0.0) / at_target[name], name))
    return best
