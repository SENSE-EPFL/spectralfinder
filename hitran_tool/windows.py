"""Candidate spectral-window search and ranking.

One function, find_combined_windows(): finds window(s) where EVERY target
gas has a detectable, mutually distinguishable line -- for a single
instrument setting/measurement that sees several gases at once (the
"simultaneous CH4/CO2/N2O detection" case from the project brief).

Only the target gases themselves are modelled: every PAIR of target species
must have their strongest in-window line >= n_separation * max(FWHM) apart,
or their signals can't be attributed to either gas. There is no separate
"background interferer" concept -- if a species matters, it belongs in
`targets`.

See find_combined_windows()'s own docstring for the full algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .broadening import doppler_fwhm, lorentz_fwhm, voigt_fwhm
from .lines import ensure_lines, line_arrays
from .spectrum import absorption_coefficient, optical_depth
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
    L_cm: float  # KNOWN, fixed path length of the actual instrument (cm) -- not solved for
    T: float = 296.0  # K
    P: float = 1.0  # atm
    s_cutoff: float = 1e-26  # cm^-1 / (molecule * cm^-2), HITRAN 'sw' units, at T
    delta_sigma: float | None = None  # window width, cm^-1; auto if None
    n_separation: float = 5.0  # required gap between targets = n_separation * max(FWHM); closer windows are rejected
    tau_min: float = 0.0  # acceptable optical depth range -ln(I/I0) at L_cm (0-40% by default): below
    tau_max: float = 0.40  # tau_min = too weak/noisy, above tau_max = saturated. Flags CombinedWindowResult.tau_ok, does not filter.
    max_windows: int = 10
    wavenumber_step: float = 0.01  # cm^-1, spectral grid resolution


@dataclass
class CombinedWindowResult:
    """One window where EVERY target species has a detectable, mutually
    distinguishable line -- for a single instrument setting that sees several
    gases at once."""
    center: float
    lo: float
    hi: float
    peak_alpha: dict  # {species_name: peak alpha in window, cm^-1}
    line_position: dict  # {species_name: position of its strongest in-window line, cm^-1}
    fwhm: dict  # {species_name: Voigt FWHM of that line, cm^-1}
    worst_pair: tuple | None  # (name_a, name_b, separation, required) -- the tightest pair; None if
    # only one target was searched (no pair to check). Diagnostic only: every window here already
    # passed the separation check, so this isn't a pass/fail flag.
    tau: dict  # {species_name: optical depth -ln(I/I0) = peak_alpha * L_cm, at the instrument's real L}
    tau_ok: dict  # {species_name: True if tau_min <= tau <= tau_max}
    score: float = 0.0


@dataclass
class CombinedSearchResult:
    """Return value of find_combined_windows(): the ranked windows plus the
    underlying per-species spectra (needed for per-window plotting)."""
    windows: list
    nu_grid: np.ndarray
    species_spectra: dict  # {target species_name: alpha array, cm^-1}


def _line_fwhm(m_id: int, iso_id: int, nu0: float, gamma_air: float, gamma_self: float,
               n_air: float, T: float, P: float, self_fraction: float) -> float:
    f_d = doppler_fwhm(nu0, m_id, iso_id, T)
    f_l = lorentz_fwhm(gamma_air, gamma_self, n_air, T, P, self_fraction)
    return voigt_fwhm(f_l, f_d)


def find_combined_windows(targets: list, config: SearchConfig) -> CombinedSearchResult:
    """Find windows where EVERY target has a detectable line, all mutually
    distinguishable -- for a single instrument setting that measures several
    gases at once in one scan (the actual "simultaneous CH4/CO2/N2O
    detection" case from the project brief: one measurement, every target
    gas visible in it).

    Algorithm:
      0. Candidate centers = local maxima of min(alpha_1, alpha_2, ..., alpha_N)
         across all targets' spectra (the "weakest signal, as strong as it can
         be" envelope) -- not each target's own line positions, which would
         bias the search toward one species and could miss the true best
         joint location sitting between individual line centers.
      1. For each candidate, find EACH target's strongest line within a
         generous search radius of that center (this radius is only used to
         identify which line is responsible -- it is NOT the final window).
      2. For every PAIR of targets, check those line positions are
         >= n_separation * max(FWHM) apart.
      3. Size the ACTUAL window AFTER finding the real peak positions: tight
         around [leftmost peak, rightmost peak] plus a small margin, unless
         config.delta_sigma is set explicitly (then that fixed width is used
         instead). This is what keeps windows from being uniformly oversized
         for a worst-case spread that most candidates don't actually have --
         then each species' peak is re-measured inside this tight window.
      4. Convert each species' peak to an optical depth tau = peak * L_cm at
         the instrument's real, fixed path length (config.L_cm), and flag
         tau_ok per species (within [tau_min, tau_max], 0-40% by default).
         This is reported, not filtered on -- see CombinedWindowResult.tau_ok.
      5. score = weakest target peak. "Weakest peak" rewards windows where
         all targets are comparably strong, not just one dominant one.

    Returns a CombinedSearchResult: .windows is the ranked list of
    CombinedWindowResult (best first, up to max_windows), plus the
    underlying spectra for plotting.
    """
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

    # Window margin: how much extra room to show past the outermost peak on
    # each side, in units of that peak's own FWHM. Only used when
    # config.delta_sigma is None (auto/tight sizing, see below).
    window_margin_factor = 0.5

    # A generous radius used ONLY to identify, per species, which line is
    # responsible near a candidate -- deliberately wide (same order as the
    # old single global window formula) so a real line never gets missed.
    # This is NOT the reported/measured window size; that's built tight
    # around the positions this search finds, further down.
    sample_all = []
    for d in species.values():
        t = d["target"]
        idx = np.nonzero(d["strong"])[0][:200]  # cap for speed on huge line lists
        sample_all.extend(
            _line_fwhm(d["m_id"], t.iso_id, d["lines"]["nu"][k], d["lines"]["gamma_air"][k],
                       d["lines"]["gamma_self"][k], d["lines"]["n_air"][k], config.T, config.P, t.mole_fraction)
            for k in idx
        )
    typical_fwhm = float(np.median(sample_all)) if sample_all else 0.1
    search_radius = 0.75 * config.n_separation * len(targets) * typical_fwhm

    candidates = []
    for center in all_centers:
        # For each species: the strongest line within `search_radius` of the
        # candidate (same proxy as before for "which line is responsible"),
        # but the WINDOW used for reporting and for measuring the actual
        # peak is built tight around these real positions afterward (below)
        # instead of reusing this generous search radius as the window --
        # that reuse is what made every window uniformly oversized.
        peak_positions, fwhm = {}, {}
        ok = True
        for name, d in species.items():
            idx_strong = np.nonzero(d["strong"])[0]
            nus = d["lines"]["nu"][idx_strong]
            near = idx_strong[(nus >= center - search_radius) & (nus <= center + search_radius)]
            if near.size == 0:
                ok = False
                break
            sws = d["lines"]["sw"][near]
            k = near[int(np.argmax(sws))]
            nu0 = float(d["lines"]["nu"][k])
            peak_positions[name] = nu0
            fwhm[name] = _line_fwhm(
                d["m_id"], d["target"].iso_id, nu0, d["lines"]["gamma_air"][k],
                d["lines"]["gamma_self"][k], d["lines"]["n_air"][k],
                config.T, config.P, d["target"].mole_fraction,
            )
        if not ok:
            continue

        names = list(peak_positions)
        separation_ok = True
        worst_pair, worst_margin = None, float("inf")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                sep = abs(peak_positions[a] - peak_positions[b])
                required = config.n_separation * max(fwhm[a], fwhm[b])
                if sep < required:
                    separation_ok = False
                margin = sep - required
                if margin < worst_margin:
                    worst_margin, worst_pair = margin, (a, b, sep, required)

        if not separation_ok:
            continue

        if config.delta_sigma is not None:
            lo, hi = center - config.delta_sigma / 2, center + config.delta_sigma / 2
        else:
            leftmost = min(peak_positions, key=peak_positions.get)
            rightmost = max(peak_positions, key=peak_positions.get)
            lo = peak_positions[leftmost] - window_margin_factor * fwhm[leftmost]
            hi = peak_positions[rightmost] + window_margin_factor * fwhm[rightmost]

        # Measure the actual peak within this (now tight) window -- built
        # around real line positions with margin, so it reliably contains
        # each species' true summit.
        i_lo, i_hi = np.searchsorted(nu_grid, (lo, hi))
        if i_hi <= i_lo:
            continue
        peak_alpha = {}
        ok = True
        for name, d in species.items():
            peak_here = float(d["alpha"][i_lo:i_hi].max())
            if peak_here <= 0:
                ok = False
                break
            peak_alpha[name] = peak_here
        if not ok:
            continue

        weakest_peak = min(peak_alpha.values())

        # With a KNOWN, fixed L (the real instrument, not something still
        # being chosen), tau = peak * L tells you whether THIS L gives a
        # usable signal here: too low means buried in noise, too high means
        # saturated. Reported per species, NOT filtered on: unlike
        # separation (a hard yes/no -- can these gases even be told apart),
        # tau is a matter of degree, and a hard reject here gives zero
        # information back when nothing clears the target band -- no sense
        # of "how close". Better to always return the strongest candidates
        # and let tau_ok show how far off they are.
        tau = {name: optical_depth(p, config.L_cm) for name, p in peak_alpha.items()}
        tau_ok = {name: config.tau_min <= t <= config.tau_max for name, t in tau.items()}

        score = weakest_peak

        candidates.append(CombinedWindowResult(
            center=float(center), lo=float(lo), hi=float(hi),
            peak_alpha=peak_alpha, line_position=peak_positions, fwhm=fwhm,
            worst_pair=worst_pair,
            tau=tau, tau_ok=tau_ok,
            score=score,
        ))

    if candidates:
        best_weakest = max(min(c.peak_alpha.values()) for c in candidates)
        candidates = [c for c in candidates if min(c.peak_alpha.values()) >= 1e-3 * best_weakest]

    candidates.sort(key=lambda c: c.score, reverse=True)

    selected = []
    for c in candidates:
        # Each candidate's own width as the exclusion radius, since windows
        # are no longer a uniform global size.
        if any(abs(c.center - s.center) < (c.hi - c.lo) for s in selected):
            continue
        selected.append(c)
        if len(selected) >= config.max_windows:
            break

    species_spectra = {name: d["alpha"] for name, d in species.items()}
    return CombinedSearchResult(selected, nu_grid, species_spectra)
