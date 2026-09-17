"""Candidate spectral-window search and ranking.

Two DIFFERENT concepts are both computed for every candidate window, from the
same `interferers` dict, and it's easy to blur them together -- keep them
apart when reading/editing this file:

  INTERFERENCE  (how much a species degrades the SNR of the measurement)
    - computed from the continuous absorption-coefficient SPECTRA (alpha vs.
      sigma): how tall is the summed interferer curve, at its peak inside the
      window, relative to the target's peak.
    - fields: interference_peak_alpha, interference_index, dominant_interferer.
    - "how much noise is riding on top of my signal in this window".

  PEAK SEPARATION  (whether two distinct peaks can be told apart at all)
    - computed from individual line POSITIONS (nu, a single number per line)
      and their widths (FWHM): is the nearest other line's center at least
      n_separation * FWHM away from this window's center.
    - fields: nearest_interferer_line, separation, required_separation, separation_ok.
    - "are these two peaks even distinguishable, independent of how strong
      either one is".

  A species can score badly on one and fine on the other -- e.g. a very
  strong line far away in wavenumber can dominate interference_index (its
  wing still reaches into the window) while easily passing separation_ok
  (its center is nowhere near). Both are reported per window; see
  find_windows_multi() for why "interferers" also has to carry sibling
  TARGET species for the separation side of this.

Pipeline for find_windows():
  1. Fetch/cache target + interferer line lists over [sigma_min, sigma_max].
  2. Compute the full-range absorption-coefficient spectrum for the target and
     for the summed interferers ONCE (not per-window -- much cheaper).
  3. Candidate window centers = every target line whose T-corrected intensity
     clears S_cutoff ("window_strategy='peak_centered'"; 'tiled' slides a
     fixed-width window across the whole range instead, ignoring where the
     target actually absorbs).
  4. For each candidate: the interference metrics and the peak-separation
     check described above, plus a suggested optical path length from the
     target optical-depth criterion.
  5. Rank by score, drop windows that duplicate an already-picked feature,
     keep the top `max_windows`, and (only for those) compute a cheap
     T/P sensitivity by finite difference.

This is a first-draft heuristic: the score and the T/P-sensitivity/interference
weighting are deliberately simple and meant to be refined once the project's
sensitivity/interference objectives (explicitly deferred for now) are
formalised.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .broadening import doppler_fwhm, lorentz_fwhm, voigt_fwhm
from .lines import ensure_lines, line_arrays
from .spectrum import absorption_coefficient, path_length_for_tau
from .species import molecule_id


@dataclass
class TargetSpecies:
    name: str
    mole_fraction: float  # 0-1; use ppm_to_fraction()/percent_to_fraction() to convert
    iso_id: int = 1


@dataclass
class SearchConfig:
    sigma_min: float  # cm^-1
    sigma_max: float  # cm^-1
    T: float = 296.0  # K
    P: float = 1.0  # atm
    s_cutoff: float = 1e-26  # cm^-1 / (molecule * cm^-2), HITRAN 'sw' units, at T
    delta_sigma: float | None = None  # window width, cm^-1; auto if None
    n_separation: float = 5.0  # [peak separation] required gap to nearest line = n_separation * max(FWHM)
    hard_separation_filter: bool = True  # [peak separation] True: reject too-close windows; False: penalize score instead
    tau_target: float = 0.10  # target optical depth -ln(I/I0)
    tau_tolerance: float = 0.30  # relative, +/- around tau_target (reported, not enforced)
    max_windows: int = 10
    wavenumber_step: float = 0.01  # cm^-1, spectral grid resolution
    window_strategy: str = "peak_centered"  # or "tiled"
    tiled_step: float | None = None  # only used for window_strategy='tiled'


@dataclass
class WindowResult:
    center: float
    lo: float
    hi: float
    target_peak_alpha: float  # cm^-1
    target_fwhm_voigt: float  # cm^-1
    n_target_lines: int

    # --- interference: SNR impact of the summed interferer spectrum (see module docstring) ---
    interference_peak_alpha: float  # cm^-1
    interference_index: float  # interference_peak_alpha / target_peak_alpha
    dominant_interferer: str | None  # species with the tallest curve in-window, if any

    # --- peak separation: can this window's line even be told apart from its neighbour (see module docstring) ---
    nearest_interferer_line: float | None  # cm^-1, position of the closest other line (target or interferer)
    separation: float | None  # cm^-1, |center - nearest_interferer_line|
    required_separation: float  # cm^-1, n_separation * max(FWHM)
    separation_ok: bool  # False if hard_separation_filter would have rejected this window

    suggested_path_length_cm: float
    sensitivity_T_percent_per_10K: float | None = None
    sensitivity_P_percent_per_0p1atm: float | None = None
    score: float = 0.0


@dataclass
class SearchResult:
    windows: list
    nu_grid: np.ndarray
    alpha_target: np.ndarray
    alpha_interference_total: np.ndarray
    interferer_spectra: dict = field(default_factory=dict)


@dataclass
class CombinedWindowResult:
    """One window where EVERY target species has a detectable, mutually
    distinguishable line -- for a single instrument setting that sees several
    gases at once (as opposed to find_windows_multi's one-setting-per-gas)."""
    center: float
    lo: float
    hi: float
    peak_alpha: dict  # {species_name: peak alpha in window, cm^-1}
    line_position: dict  # {species_name: position of its strongest in-window line, cm^-1}
    fwhm: dict  # {species_name: Voigt FWHM of that line, cm^-1}
    worst_pair: tuple | None  # (name_a, name_b, separation, required) -- the tightest pair
    separation_ok: bool  # True if EVERY pair of targets clears n_separation * max(FWHM)
    background_peak_alpha: float  # cm^-1, peak of the summed background spectrum in-window
    background_interference_index: float  # background_peak_alpha / weakest target peak
    dominant_background: str | None
    suggested_path_length_cm: dict  # {species_name: L to reach tau_target for THAT species alone}
    suggested_path_length_cm_combined: float  # max of the above -- safe choice covering all species
    score: float = 0.0


def _auto_delta_sigma(fwhm_values: np.ndarray, sigma_min: float, sigma_max: float) -> float:
    typical = float(np.median(fwhm_values)) if fwhm_values.size else 0.1
    span = sigma_max - sigma_min
    return float(np.clip(20 * typical, 0.5, span / 10))


def _line_fwhm(m_id: int, iso_id: int, nu0: float, gamma_air: float, gamma_self: float,
               n_air: float, T: float, P: float, self_fraction: float) -> float:
    f_d = doppler_fwhm(nu0, m_id, iso_id, T)
    f_l = lorentz_fwhm(gamma_air, gamma_self, n_air, T, P, self_fraction)
    return voigt_fwhm(f_l, f_d)


def find_windows(target: TargetSpecies, interferers: dict, config: SearchConfig) -> SearchResult:
    m_id = molecule_id(target.name)

    table = ensure_lines(target.name, config.sigma_min, config.sigma_max, target.iso_id)
    lines_t = line_arrays(table, config.sigma_min, config.sigma_max, config.T)
    strong = lines_t["sw"] >= config.s_cutoff

    nu_grid, alpha_target = absorption_coefficient(
        table, m_id, target.iso_id, target.mole_fraction,
        config.sigma_min, config.sigma_max, config.T, config.P,
        config.wavenumber_step, config.s_cutoff,
    )

    interferer_spectra = {}
    interferer_lines = {}
    alpha_interference_total = np.zeros_like(alpha_target)
    for name, frac in interferers.items():
        try:
            itable = ensure_lines(name, config.sigma_min, config.sigma_max)
        except Exception:
            continue  # no HITRAN data available for this species/range; treat as no interference
        i_id = molecule_id(name)
        ilines = line_arrays(itable, config.sigma_min, config.sigma_max, config.T)
        i_strong = ilines["sw"] >= config.s_cutoff
        interferer_lines[name] = {k: v[i_strong] for k, v in ilines.items()}

        _, alpha_i = absorption_coefficient(
            itable, i_id, 1, frac,
            config.sigma_min, config.sigma_max, config.T, config.P,
            config.wavenumber_step, config.s_cutoff,
        )
        interferer_spectra[name] = alpha_i
        alpha_interference_total += alpha_i

    if not np.any(strong):
        return SearchResult([], nu_grid, alpha_target, alpha_interference_total, interferer_spectra)

    fwhm_strong = np.array([
        _line_fwhm(m_id, target.iso_id, nu0, ga, gs, na, config.T, config.P, target.mole_fraction)
        for nu0, ga, gs, na in zip(
            lines_t["nu"][strong], lines_t["gamma_air"][strong],
            lines_t["gamma_self"][strong], lines_t["n_air"][strong],
        )
    ])

    delta_sigma = config.delta_sigma or _auto_delta_sigma(fwhm_strong, config.sigma_min, config.sigma_max)

    if config.window_strategy == "tiled":
        step = config.tiled_step or delta_sigma
        centers = np.arange(config.sigma_min + delta_sigma / 2, config.sigma_max, step)
    else:
        centers = np.sort(lines_t["nu"][strong])

    candidates = []
    idx_strong = np.nonzero(strong)[0]
    for center in centers:
        lo, hi = center - delta_sigma / 2, center + delta_sigma / 2
        # nu_grid is sorted -> binary search for the window's index range instead
        # of a full-array boolean mask (this loop runs once per candidate line,
        # which can be tens of thousands of times over a wide search range).
        i_lo, i_hi = np.searchsorted(nu_grid, (lo, hi))
        if i_hi <= i_lo:
            continue

        peak_t = float(alpha_target[i_lo:i_hi].max())
        if peak_t <= 0:
            continue

        # --- interference: peak of the summed interferer SPECTRUM inside this window ---
        peak_i = float(alpha_interference_total[i_lo:i_hi].max())
        interference_index = peak_i / peak_t if peak_t > 0 else float("inf")

        dominant = None
        if peak_i > 0 and interferer_spectra:
            dominant = max(interferer_spectra, key=lambda n: interferer_spectra[n][i_lo:i_hi].max())

        # --- peak separation: nearest interferer LINE POSITION, regardless of its strength ---
        in_window = (lines_t["nu"][idx_strong] >= lo) & (lines_t["nu"][idx_strong] <= hi)
        n_target_lines = int(in_window.sum())
        window_line_idx = idx_strong[in_window]
        fwhm_here = max(
            (_line_fwhm(m_id, target.iso_id, lines_t["nu"][k], lines_t["gamma_air"][k],
                        lines_t["gamma_self"][k], lines_t["n_air"][k], config.T, config.P,
                        target.mole_fraction) for k in window_line_idx),
            default=float(np.median(fwhm_strong)),
        )

        nearest_line, separation, name_at_nearest, fwhm_nearest = None, None, None, 0.0
        for name, ilines in interferer_lines.items():
            if ilines["nu"].size == 0:
                continue
            j = int(np.argmin(np.abs(ilines["nu"] - center)))
            d = abs(ilines["nu"][j] - center)
            if separation is None or d < separation:
                separation = d
                nearest_line = float(ilines["nu"][j])
                name_at_nearest = name
                fwhm_nearest = _line_fwhm(
                    molecule_id(name), int(ilines["iso_id"][j]), ilines["nu"][j],
                    ilines["gamma_air"][j], ilines["gamma_self"][j], ilines["n_air"][j],
                    config.T, config.P, interferers[name],
                )

        required_separation = config.n_separation * max(fwhm_here, fwhm_nearest)
        separation_ok = separation is None or separation >= required_separation

        if config.hard_separation_filter and not separation_ok:
            continue

        penalty = 1.0 if separation_ok else 0.2
        score = penalty * peak_t / (1 + interference_index)

        candidates.append(WindowResult(
            center=float(center), lo=float(lo), hi=float(hi),
            target_peak_alpha=peak_t, target_fwhm_voigt=fwhm_here, n_target_lines=n_target_lines,
            interference_peak_alpha=peak_i, interference_index=interference_index,
            dominant_interferer=dominant or name_at_nearest,
            nearest_interferer_line=nearest_line, separation=separation,
            required_separation=required_separation, separation_ok=separation_ok,
            suggested_path_length_cm=path_length_for_tau(peak_t, config.tau_target),
            score=score,
        ))

    if candidates:
        peak_ref = max(c.target_peak_alpha for c in candidates)
        candidates = [c for c in candidates if c.target_peak_alpha >= 1e-3 * peak_ref]

    candidates.sort(key=lambda w: w.score, reverse=True)

    selected = []
    for w in candidates:
        if any(abs(w.center - s.center) < delta_sigma for s in selected):
            continue
        selected.append(w)
        if len(selected) >= config.max_windows:
            break

    for w in selected:
        w.sensitivity_T_percent_per_10K = _sensitivity(
            table, m_id, target, w, config, dT=10.0, dP=0.0,
        )
        w.sensitivity_P_percent_per_0p1atm = _sensitivity(
            table, m_id, target, w, config, dT=0.0, dP=0.1,
        )

    return SearchResult(selected, nu_grid, alpha_target, alpha_interference_total, interferer_spectra)


def find_windows_multi(targets: list, config: SearchConfig, background: dict | None = None) -> dict:
    """Run find_windows() once per target species, for ONE INSTRUMENT SETTING
    PER GAS -- e.g. a scanning laser that visits a different window for CH4,
    then a different one for N2O, etc. If you instead want a SINGLE window
    where all targets are visible at once (one fixed setting sees all of
    them), use find_combined_windows() below, not this function.

    Two different relationships get conflated as "interference" if handled by
    hand: (a) target vs. target -- their peaks must not sit on top of each
    other, or the signal can't be attributed to either gas; (b) target vs.
    background (H2O...) -- genuine interference that degrades the SNR without
    being something you're trying to measure. This function makes (a)
    automatic and safe: when searching windows for one target, every OTHER
    target is added to its interferer dict at ITS OWN concentration, on top
    of the shared `background`. The existing separation/interference
    machinery in find_windows() then treats sibling targets exactly like any
    other interferer, which is exactly what's needed to keep their windows
    distinct.

    Returns {target_name: SearchResult}, one run per target.
    """
    background = background or {}
    results = {}
    for target in targets:
        interferers = dict(background)
        for other in targets:
            if other.name == target.name:
                continue
            interferers[other.name] = other.mole_fraction
        results[target.name] = find_windows(target, interferers, config)
    return results


def find_combined_windows(targets: list, config: SearchConfig, background: dict | None = None) -> list:
    """Find windows where EVERY target has a detectable line, all mutually
    distinguishable -- for a single instrument setting that measures several
    gases at once in one scan (the actual "simultaneous CH4/CO2/N2O
    detection" case from the project brief). See find_windows_multi's
    docstring for how this differs from the one-window-per-gas case.

    Algorithm:
      0. Candidate centers = local maxima of min(alpha_1, alpha_2, ..., alpha_N)
         across all targets' spectra (the "weakest signal, as strong as it can
         be" envelope) -- not each target's own line positions, which would
         bias the search toward one species and could miss the true best
         joint location sitting between individual line centers.
      1. Reject the window unless EVERY target has a peak > 0 inside it.
      2. For every PAIR of targets, check their strongest in-window lines are
         >= n_separation * max(FWHM) apart (same rule as find_windows, but
         applied between the co-located targets instead of against a
         species you're trying to avoid).
      3. background (if given) is scored as genuine interference against the
         WEAKEST target in the window (the one most vulnerable to it).
      4. score = weakest target peak / (1 + background_interference_index),
         penalized if any pair fails separation and hard_separation_filter
         is off. "Weakest peak" rewards windows where all targets are
         comparably strong, not just one dominant one.

    Returns a list of CombinedWindowResult, best first, up to max_windows.
    """
    background = background or {}

    species = {}
    for t in targets:
        table = ensure_lines(t.name, config.sigma_min, config.sigma_max, t.iso_id)
        lines_ = line_arrays(table, config.sigma_min, config.sigma_max, config.T)
        strong = lines_["sw"] >= config.s_cutoff
        m_id = molecule_id(t.name)
        nu_grid, alpha = absorption_coefficient(
            table, m_id, t.iso_id, t.mole_fraction,
            config.sigma_min, config.sigma_max, config.T, config.P,
            config.wavenumber_step, config.s_cutoff,
        )
        species[t.name] = dict(target=t, m_id=m_id, lines=lines_, strong=strong,
                                nu_grid=nu_grid, alpha=alpha)

    nu_grid = next(iter(species.values()))["nu_grid"]
    bg_spectra = {}
    bg_total = np.zeros_like(nu_grid)
    for name, frac in background.items():
        try:
            btable = ensure_lines(name, config.sigma_min, config.sigma_max)
        except Exception:
            continue
        _, a = absorption_coefficient(
            btable, molecule_id(name), 1, frac,
            config.sigma_min, config.sigma_max, config.T, config.P,
            config.wavenumber_step, config.s_cutoff,
        )
        bg_spectra[name] = a
        bg_total += a

    # Candidate centers = local maxima of the MIN-envelope across all target
    # spectra, not each species' own line positions. Anchoring candidates on
    # one species' lines (as an earlier version of this function did) biases
    # the search: the true best "all three at once" location can sit BETWEEN
    # individual line centers, and a one-species anchor never probes that.
    # The min-envelope directly answers "where is the weakest of the three
    # signals as strong as it can possibly be" -- exactly the score this
    # function ranks by -- and is a single vectorized pass over the shared
    # grid rather than a loop over every strong line of every species.
    envelope = np.minimum.reduce([d["alpha"] for d in species.values()])
    is_local_max = np.zeros(envelope.shape, dtype=bool)
    is_local_max[1:-1] = (envelope[1:-1] > envelope[:-2]) & (envelope[1:-1] >= envelope[2:]) & (envelope[1:-1] > 0)
    peak_idx = np.nonzero(is_local_max)[0]
    all_centers = nu_grid[peak_idx[np.argsort(envelope[peak_idx])[::-1]]]

    if config.delta_sigma is not None:
        delta_sigma = config.delta_sigma
    else:
        sample_fwhm = []
        for d in species.values():
            t = d["target"]
            idx = np.nonzero(d["strong"])[0][:200]  # cap for speed on huge line lists
            for k in idx:
                sample_fwhm.append(_line_fwhm(
                    d["m_id"], t.iso_id, d["lines"]["nu"][k], d["lines"]["gamma_air"][k],
                    d["lines"]["gamma_self"][k], d["lines"]["n_air"][k], config.T, config.P, t.mole_fraction,
                ))
        typical = float(np.median(sample_fwhm)) if sample_fwhm else 0.1
        span = config.sigma_max - config.sigma_min
        # combined windows need room for len(targets) peaks, not just one
        delta_sigma = float(np.clip(20 * typical * len(targets), 0.5, span / 10))

    candidates = []
    for center in all_centers:
        lo, hi = center - delta_sigma / 2, center + delta_sigma / 2
        # All species share the same nu_grid (same sigma_min/max/step), so the
        # index range only needs computing once per candidate, via binary
        # search on the sorted grid -- not a full-array mask per species.
        i_lo, i_hi = np.searchsorted(nu_grid, (lo, hi))
        if i_hi <= i_lo:
            continue

        peak_alpha, line_position, fwhm = {}, {}, {}
        complete = True
        for name, d in species.items():
            peak_here = float(d["alpha"][i_lo:i_hi].max())
            if peak_here <= 0:
                complete = False
                break
            idx_strong = np.nonzero(d["strong"])[0]
            nus = d["lines"]["nu"][idx_strong]
            in_win = (nus >= lo) & (nus <= hi)
            if not np.any(in_win):
                complete = False
                break
            sws = d["lines"]["sw"][idx_strong][in_win]
            k = idx_strong[in_win][int(np.argmax(sws))]  # strongest in-window line for this species
            nu0 = float(d["lines"]["nu"][k])
            peak_alpha[name] = peak_here
            line_position[name] = nu0
            fwhm[name] = _line_fwhm(
                d["m_id"], d["target"].iso_id, nu0, d["lines"]["gamma_air"][k],
                d["lines"]["gamma_self"][k], d["lines"]["n_air"][k],
                config.T, config.P, d["target"].mole_fraction,
            )
        if not complete:
            continue  # this window doesn't have all targets present -- not a combined candidate

        names = list(line_position)
        separation_ok = True
        worst_pair, worst_margin = None, float("inf")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                sep = abs(line_position[a] - line_position[b])
                required = config.n_separation * max(fwhm[a], fwhm[b])
                if sep < required:
                    separation_ok = False
                margin = sep - required
                if margin < worst_margin:
                    worst_margin, worst_pair = margin, (a, b, sep, required)

        if config.hard_separation_filter and not separation_ok:
            continue

        peak_bg = float(bg_total[i_lo:i_hi].max()) if bg_spectra else 0.0
        weakest_name = min(peak_alpha, key=peak_alpha.get)
        weakest_peak = peak_alpha[weakest_name]
        bg_index = peak_bg / weakest_peak if weakest_peak > 0 else float("inf")
        dominant_bg = (max(bg_spectra, key=lambda n: bg_spectra[n][i_lo:i_hi].max())
                        if peak_bg > 0 and bg_spectra else None)

        path_lengths = {name: path_length_for_tau(p, config.tau_target) for name, p in peak_alpha.items()}
        finite = [L for L in path_lengths.values() if L != float("inf")]
        combined_L = max(finite) if finite else float("inf")

        penalty = 1.0 if separation_ok else 0.2
        score = penalty * weakest_peak / (1 + bg_index)

        candidates.append(CombinedWindowResult(
            center=float(center), lo=float(lo), hi=float(hi),
            peak_alpha=peak_alpha, line_position=line_position, fwhm=fwhm,
            worst_pair=worst_pair, separation_ok=separation_ok,
            background_peak_alpha=peak_bg, background_interference_index=bg_index,
            dominant_background=dominant_bg,
            suggested_path_length_cm=path_lengths, suggested_path_length_cm_combined=combined_L,
            score=score,
        ))

    if candidates:
        best_weakest = max(min(c.peak_alpha.values()) for c in candidates)
        candidates = [c for c in candidates if min(c.peak_alpha.values()) >= 1e-3 * best_weakest]

    candidates.sort(key=lambda c: c.score, reverse=True)

    selected = []
    for c in candidates:
        if any(abs(c.center - s.center) < delta_sigma for s in selected):
            continue
        selected.append(c)
        if len(selected) >= config.max_windows:
            break

    return selected


def _sensitivity(table, m_id, target: TargetSpecies, w: WindowResult, config: SearchConfig,
                  dT: float, dP: float) -> float:
    _, alpha_pert = absorption_coefficient(
        table, m_id, target.iso_id, target.mole_fraction,
        w.lo, w.hi, config.T + dT, config.P + dP,
        config.wavenumber_step, config.s_cutoff,
    )
    peak_pert = float(alpha_pert.max()) if alpha_pert.size else 0.0
    if w.target_peak_alpha <= 0:
        return 0.0
    return 100.0 * (peak_pert - w.target_peak_alpha) / w.target_peak_alpha
