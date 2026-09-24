"""Candidate spectral-window search and ranking, in TWO STAGES (per the
project supervisor's proposed workflow):

  STAGE 1 -- fast/coarse candidate search, over the FULL search range.
    Uses only line POSITIONS and a cheap intensity PROXY (line strength S(T)
    times number density, no line shape at all) -- never the expensive full
    Voigt-profile spectrum. This is what makes it fast enough to scan a wide
    range (thousands of cm^-1): no HAPI absorptionCoefficient_* call happens
    in this stage.

  STAGE 2 -- refine, only on the small number of candidates stage 1 keeps.
    NOW compute the real absorption spectrum (Doppler + collisional
    broadening, exact HAPI Voigt profile) but only over each candidate's
    narrow window (delta_sigma, e.g. 0.3-0.4 cm^-1 -- a DFB laser's current-
    scan range), which is cheap precisely because the range is tiny. Convert
    to a real peak alpha (cm^-1), then to transmission = exp(-alpha * L_cm)
    at the instrument's real, fixed path length, and flag whether that falls
    in the acceptable range.

Doing it the other way around -- full-width, full-physics spectra over the
whole search range before even looking for candidates -- is what made the
single-stage version of this function slow (tens of seconds to minutes):
HAPI's per-line Python loop pays for every line in range regardless of how
much of that range is ever a plausible answer.

Only the target gases themselves are modelled: every PAIR of target species
must have their strongest in-window line >= n_separation * max(FWHM) apart,
or their signals can't be attributed to either gas. There is no separate
"background interferer" concept -- if a species matters, it belongs in
`targets`.

See find_combined_windows()'s own docstring for the full algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log10

import numpy as np

from .broadening import doppler_fwhm, lorentz_fwhm, voigt_fwhm
from .lines import ensure_lines, line_arrays
from .spectrum import absorption_coefficient, number_density, optical_depth
from .species import molecule_id

# Stage 2 hands HAPI a narrow, already-selected window -- this just keeps numerically-zero
# lines out of its sum, it's not a presence decision (that's SearchConfig.s_relative_floor).
_HAPI_INTENSITY_FLOOR = 1e-30


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
    delta_sigma: float = 0.35  # window width, cm^-1 -- a real hardware constraint (e.g. a DFB
    # laser's current-tuning range is typically ~0.3-0.55 cm^-1), NOT auto-derived
    T: float = 296.0  # K
    P: float = 1.0  # atm
    # A line counts as "strong" (candidate-generating) if its PROXY intensity
    # (sw * n_total(P,T) * mole_fraction, same quantity as stage 1's ranking proxy -- see
    # find_combined_windows' docstring) is >= proxy_cutoff. Filtering on the proxy rather than
    # on raw HITRAN sw puts every species on the same physical footing regardless of
    # concentration: a raw-sw cutoff shared across a 1 ppm trace gas and 30% H2O isn't
    # comparable (a "weak" H2O line still out-absorbs a "strong" N2O one), and comparing each
    # line to that SAME SPECIES' own strongest line elsewhere in the range doesn't work either
    # -- CH4/N2O have much stronger fundamental bands elsewhere in 2000-9000 cm^-1 than in a
    # combination-band region like 2270 or 2570 nm, which made that region look artificially
    # weak by comparison and vanish from the results entirely.
    proxy_cutoff: float = 1e-12  # cm^-2, same units as the stage-1 ranking proxy
    n_separation: float = 3.0  # required gap between targets = n_separation * max(FWHM); closer windows are rejected
    # Acceptable I/I0 = exp(-alpha*L_cm) range: below transmission_min = too strong/saturated
    # absorption, above transmission_max = too weak/undetectable. I/I0 is HIGH for a weak
    # absorption and LOW for a strong one -- targeting e.g. 5-20% signal attenuation means
    # I/I0 in [0.80, 0.95] (= 1-0.20, 1-0.05), NOT [0.05, 0.20].
    transmission_min: float = 0.80
    transmission_max: float = 0.95
    # Flags CombinedWindowResult.transmission_ok, does not filter.
    max_windows: int = 10
    max_stage1_candidates: int = 200  # cap on how many stage-1 candidates get the expensive stage-2 treatment
    wavenumber_step: float = 0.01  # cm^-1, spectral grid resolution used in stage 2


@dataclass
class CombinedWindowResult:
    """One window where EVERY target species has a detectable, mutually
    distinguishable line -- for a single instrument setting that sees several
    gases at once."""
    center: float
    lo: float
    hi: float
    peak_alpha: dict  # {species_name: peak alpha in window, cm^-1} -- real Voigt profile (stage 2)
    line_position: dict  # {species_name: position of its strongest in-window line, cm^-1}
    fwhm: dict  # {species_name: Voigt FWHM of that line, cm^-1}
    worst_pair: tuple | None  # (name_a, name_b, separation, required) -- the tightest pair; None if
    # only one target was searched (no pair to check). Diagnostic only: every window here already
    # passed the separation check, so this isn't a pass/fail flag.
    tau: dict  # {species_name: optical depth -ln(I/I0) = peak_alpha * L_cm, at the instrument's real L}
    transmission: dict  # {species_name: I/I0 = exp(-tau)}
    transmission_ok: dict  # {species_name: True if transmission_min <= transmission <= transmission_max}
    score: float = 0.0  # weakest real peak_alpha across species -- diagnostic only, NOT the
    # ranking key: windows are ranked by _rank_key() (species in transmission range first,
    # then distance-from-range), not by raw peak strength alone.


@dataclass
class CombinedSearchResult:
    """Return value of find_combined_windows(): the ranked windows plus, for
    plotting, each window's OWN narrow stage-2 spectra -- there's no single
    shared spectrum any more (unlike the old single-stage version): every
    window has its own tiny [lo, hi] range, computed independently.
    `window_spectra[i]` is a {species_name: (nu_grid, alpha)} dict matching
    `windows[i]`."""
    windows: list
    window_spectra: list


def _line_fwhm(m_id: int, iso_id: int, nu0: float, gamma_air: float, gamma_self: float,
               n_air: float, T: float, P: float, self_fraction: float) -> float:
    f_d = doppler_fwhm(nu0, m_id, iso_id, T)
    f_l = lorentz_fwhm(gamma_air, gamma_self, n_air, T, P, self_fraction)
    return voigt_fwhm(f_l, f_d)


def _range_penalty(tr: float, tmin: float, tmax: float) -> float:
    """0 if tr is inside [tmin, tmax], else its log10 distance to the nearest edge.
    Log-space because transmission/absorption is inherently exponential -- being 10x too
    weak and 10x too strong should count as an equally bad miss."""
    if tr < tmin:
        # tr can underflow to exactly 0.0 for a very saturated line (tau so large that
        # exp(-tau) rounds to 0 in float64) -- log10(0) would raise, so floor it first.
        return log10(tmin) - log10(max(tr, 1e-300))
    if tr > tmax:
        return log10(tr) - log10(tmax)
    return 0.0


def _rank_key(c: CombinedWindowResult, tmin: float, tmax: float) -> tuple:
    """Final Stage-2 ranking: how many species land in the acceptable transmission
    range comes first (a window good for 3/3 gases beats one good for 1/3, even if
    that one has a stronger raw peak); the total distance-from-range of whichever
    species are still outside it breaks ties. Raw peak strength (the old sole
    criterion) no longer drives ranking on its own -- see README's Known limitations
    for why that mattered (the validated case ranked #3, not #1, under it)."""
    n_ok = sum(c.transmission_ok.values())
    penalty = sum(_range_penalty(tr, tmin, tmax) for tr in c.transmission.values())
    return (n_ok, -penalty)


def find_combined_windows(targets: list, config: SearchConfig) -> CombinedSearchResult:
    """Find windows where EVERY target has a detectable line, all mutually
    distinguishable -- for a single instrument setting that measures several
    gases at once in one scan (the actual "simultaneous CH4/CO2/N2O
    detection" case from the project brief: one measurement, every target
    gas visible in it).

    STAGE 1 (cheap, whole range):
      0. Load each target's lines (positions, T-corrected intensity S(T),
         broadening parameters) -- no spectrum computed yet.
      1. proxy_intensity per line = S(T) * n_total(P,T) * mole_fraction.
         This has units of cm^-2, NOT alpha's cm^-1 -- it deliberately skips
         the line shape (no width, no Voigt) so it's cheap to compute for
         every line. It's only used to RANK candidates against each other,
         consistently across species, never reported as a physical alpha.
      2. Candidate centers = every target's own strong-line positions.
      3. For each candidate: does EVERY species have a strong line within
         delta_sigma of it? If not, reject. Otherwise take each species'
         strongest such line (by proxy_intensity) as its representative.
      4. Pairwise separation check (>= n_separation * max(FWHM), FWHM from
         the cheap analytic formula in broadening.py -- still no full
         spectrum). Reject if any pair fails.
      5. stage1_score = weakest proxy_intensity across species. Sort, dedup
         (reject centers < delta_sigma from an already-kept one), keep up to
         max_stage1_candidates.

    STAGE 2 (expensive but now on a handful of narrow windows):
      6. For each surviving candidate, compute the REAL absorption spectrum
         (exact HAPI Voigt profile, real Doppler+collisional widths) for
         each species, but ONLY over that candidate's own
         [center - delta_sigma/2, center + delta_sigma/2] -- cheap because
         the range is tiny, not because the physics is approximated.
      7. tau = peak_alpha * L_cm (instrument's real path length);
         transmission = exp(-tau); flag transmission_ok against
         [transmission_min, transmission_max] (see CombinedWindowResult.transmission_ok).
      8. Filter out survivors whose weakest peak is negligible next to the best
         one found (< 0.1%). Rank by _rank_key: how many species land inside
         [transmission_min, transmission_max] first (a window good for all
         targets beats one good for only some, even with weaker raw peaks),
         then by total log-distance-from-range for whichever species are
         still outside it. Sort, dedup, keep top max_windows.

    Returns a CombinedSearchResult: .windows is the ranked list of
    CombinedWindowResult (best first, up to max_windows), plus the
    per-window spectra for plotting.
    """
    delta_sigma = config.delta_sigma
    n_total = number_density(config.P, config.T)

    # ============================== STAGE 1 ==============================
    species = {}
    for t in targets:
        table = ensure_lines(t.name, config.sigma_min, config.sigma_max, t.iso_id)
        lines_ = line_arrays(table, config.sigma_min, config.sigma_max, config.T)
        m_id = molecule_id(t.name)
        # proxy_intensity: S(T) * N_species, no line shape -- see module docstring.
        proxy = lines_["sw"] * (n_total * t.mole_fraction)
        strong = proxy >= config.proxy_cutoff
        species[t.name] = dict(target=t, m_id=m_id, lines=lines_, strong=strong, proxy=proxy)

    all_centers = np.sort(np.concatenate(
        [d["lines"]["nu"][d["strong"]] for d in species.values() if np.any(d["strong"])]
    ))

    stage1 = []
    for center in all_centers:
        lo, hi = center - delta_sigma / 2, center + delta_sigma / 2

        peak_positions, fwhm, proxy_val = {}, {}, {}
        complete = True
        for name, d in species.items():
            idx_strong = np.nonzero(d["strong"])[0]
            nus = d["lines"]["nu"][idx_strong]
            in_win = (nus >= lo) & (nus <= hi)
            if not np.any(in_win):
                complete = False
                break
            candidates_here = idx_strong[in_win]
            k = candidates_here[int(np.argmax(d["proxy"][candidates_here]))]
            nu0 = float(d["lines"]["nu"][k])
            peak_positions[name] = nu0
            proxy_val[name] = float(d["proxy"][k])
            fwhm[name] = _line_fwhm(
                d["m_id"], d["target"].iso_id, nu0, d["lines"]["gamma_air"][k],
                d["lines"]["gamma_self"][k], d["lines"]["n_air"][k],
                config.T, config.P, d["target"].mole_fraction,
            )
        if not complete:
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

        stage1_score = min(proxy_val.values())
        stage1.append((stage1_score, float(center), lo, hi, peak_positions, fwhm, worst_pair))

    stage1.sort(key=lambda c: c[0], reverse=True)

    stage1_selected = []
    for c in stage1:
        center = c[1]
        if any(abs(center - s[1]) < delta_sigma for s in stage1_selected):
            continue
        stage1_selected.append(c)
        if len(stage1_selected) >= config.max_stage1_candidates:
            break

    # ============================== STAGE 2 ==============================
    candidates = []
    for _, center, lo, hi, peak_positions, fwhm, worst_pair in stage1_selected:
        peak_alpha = {}
        window_spectrum = {}
        ok = True
        for name, d in species.items():
            table = ensure_lines(name, lo, hi, d["target"].iso_id)  # sub-range already covered from stage 1
            nu_grid, alpha = absorption_coefficient(
                table, d["m_id"], d["target"].iso_id, d["target"].mole_fraction,
                lo, hi, config.T, config.P, config.wavenumber_step, _HAPI_INTENSITY_FLOOR,
            )
            if alpha.size == 0 or float(alpha.max()) <= 0:
                ok = False
                break
            peak_alpha[name] = float(alpha.max())
            window_spectrum[name] = (nu_grid, alpha)
        if not ok:
            continue

        weakest_peak = min(peak_alpha.values())

        tau = {name: optical_depth(p, config.L_cm) for name, p in peak_alpha.items()}
        transmission = {name: exp(-t) for name, t in tau.items()}
        transmission_ok = {
            name: config.transmission_min <= tr <= config.transmission_max
            for name, tr in transmission.items()
        }

        result = CombinedWindowResult(
            center=center, lo=lo, hi=hi,
            peak_alpha=peak_alpha, line_position=peak_positions, fwhm=fwhm,
            worst_pair=worst_pair,
            tau=tau, transmission=transmission, transmission_ok=transmission_ok,
            score=weakest_peak,
        )
        candidates.append((result, window_spectrum))

    if candidates:
        best_weakest = max(min(c.peak_alpha.values()) for c, _ in candidates)
        candidates = [(c, spec) for c, spec in candidates if min(c.peak_alpha.values()) >= 1e-3 * best_weakest]

    candidates.sort(key=lambda cs: _rank_key(cs[0], config.transmission_min, config.transmission_max), reverse=True)

    selected, selected_spectra = [], []
    for c, spec in candidates:
        if any(abs(c.center - s.center) < delta_sigma for s in selected):
            continue
        selected.append(c)
        selected_spectra.append(spec)
        if len(selected) >= config.max_windows:
            break

    return CombinedSearchResult(selected, selected_spectra)
