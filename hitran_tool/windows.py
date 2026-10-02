"""Candidate spectral-window search and ranking, in TWO STAGES (per the
project supervisor's proposed workflow), plus per-window diagnostics:

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
    scan range), which is cheap precisely because the range is tiny. Take
    each target's peak (the top of its real curve), convert to transmission
    = exp(-alpha * L_cm) at the instrument's real, fixed path length, and
    flag whether that falls in the acceptable range.

  SATURATION FILTER + DIAGNOSTICS -- walking the ranked list until
    max_windows windows pass: background interferers (interference.py),
    a hard filter on the TOTAL transmission across the window (all species),
    and temperature sensitivity (temperature.py). The interference ratio and
    the T coefficients are reported only, never used to rank.

Doing it the other way around -- full-width, full-physics spectra over the
whole search range before even looking for candidates -- is what made the
single-stage version of this function slow (tens of seconds to minutes):
HAPI's per-line Python loop pays for every line in range regardless of how
much of that range is ever a plausible answer.

Every PAIR of targets must have their selected in-window lines
>= n_separation * max(FWHM) apart, or their signals can't be attributed to
either gas. Isotopologues are separate targets (TargetSpecies.iso_id), so
e.g. CH4 and (13C)H4 are checked against each other like any two gases.

See find_combined_windows()'s own docstring for the full algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, inf, log, log10

import numpy as np

from .broadening import doppler_fwhm, lorentz_fwhm, voigt_fwhm
from .interference import compute_interference, ensure_interferer_lines, interferer_list, target_molecules
from .lines import ensure_lines, line_arrays
from .spectrum import absorption_coefficient, number_density, optical_depth, window_grid
from .species import molecule_id, natural_abundance, species_label
from .temperature import temperature_dependence

# Stage 2 hands HAPI a narrow, already-selected window -- this just keeps numerically-zero
# lines out of its sum, it's not a presence decision (that's SearchConfig.proxy_cutoff).
_HAPI_INTENSITY_FLOOR = 1e-30

# A species' spectrum is flagged (CombinedWindowResult.peak_is_window_max = False) when
# something else in the window reaches more than this factor times its selected peak: the
# wing of a strong line just outside the window (max at the window's edge), or a taller
# top inside it that couldn't be used (it would break the separation check). Also the
# threshold for switching to that taller top at all.
_WINDOW_MAX_TOLERANCE = 1.05


@dataclass
class TargetSpecies:
    name: str  # HITRAN molecule name, e.g. 'CH4'
    mole_fraction: float  # 0-1, of the WHOLE molecule (all isotopologues); use ppm_to_fraction() etc.
    iso_id: int = 1  # HITRAN isotopologue number: 1 = principal, e.g. CH4 iso 2 = (13C)H4
    abundance: float | None = None  # this isotopologue's share of the molecule; None = natural
    # (HITRAN) abundance. Set it only to simulate a non-natural isotopic composition.

    @property
    def label(self) -> str:
        """Key used in every result dict: 'CH4' for iso 1, the HITRAN formula otherwise."""
        return species_label(self.name, self.iso_id)


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
    # Saturation filter (this one DOES filter): the TOTAL transmission the detector sees,
    # anywhere across the scanned window -- all targets + all interferers summed, e.g. the
    # wing of a saturated H2O line at the window's edge -- must stay >= this. Deliberately
    # separate from transmission_min (which only flags each target's own line): 0.5 = reject
    # windows where more than half the light is absorbed somewhere in the scan.
    window_transmission_min: float = 0.5
    # How interference enters the RANKING (it's always reported either way):
    #   'none'     -- not at all (rank = species in transmission range, then distance from it)
    #   'filter'   -- drop windows where any target's interference > interference_max
    #   'tiebreak' -- rank = species in transmission range, then LOWEST worst interference
    #   'score'    -- one score, no "species in range" first: distance from range (decades,
    #                 see _range_penalty) + interference_weight * log10(1 + worst / interference_max)
    #                 -- so interference_max costs ~0.3 decade, i.e. like being a factor 2 off range.
    # The interference used here is computed for EVERY stage-2 candidate, from the other
    # targets and the targets' own isotopologues only (not the background CSV: ~10x cheaper,
    # and it changed nothing in the validation case -- see compare_variants.py).
    rank_by_interference: str = "none"
    interference_max: float = 0.05
    interference_weight: float = 1.0
    max_windows: int = 10
    max_stage1_candidates: int = 200  # cap on how many stage-1 candidates get the expensive stage-2 treatment
    # Spectral grid step in stage 2 (cm^-1). Each peak's top is refined BETWEEN grid points
    # (parabola through log(alpha), see _refine_top), so the step mostly shapes the plotted
    # curves -- but keep it well below the narrowest FWHM (~0.008 cm^-1 at 30 mbar) so the
    # 3 points around a top stay on the peak and the measured widths stay accurate.
    wavenumber_step: float = 0.002
    # Temperature sweep (temperature.py): alpha at T, T+dT, ..., T+steps*dT, constant P.
    temperature_step_K: float = 1.0
    temperature_steps: int = 5


@dataclass
class CombinedWindowResult:
    """One window where EVERY target species has a detectable, mutually
    distinguishable line -- for a single instrument setting that sees several
    gases at once. Every dict is keyed by TargetSpecies.label."""
    center: float
    lo: float
    hi: float
    peak_position: dict  # {species: where its SELECTED peak is -- the top of its curve in the
    # window (see find_combined_windows step 7), cm^-1}
    peak_alpha: dict  # {species: alpha at that top, cm^-1 -- real Voigt spectrum (stage 2)}
    fwhm: dict  # {species: width of that peak at half height, measured on the curve, cm^-1}
    worst_pair: tuple | None  # (name_a, name_b, separation, required) -- the tightest pair; None if
    # only one target was searched (no pair to check). Diagnostic only: every window here already
    # passed the separation check, so this isn't a pass/fail flag.
    tau: dict  # {species: optical depth -ln(I/I0) = peak_alpha * L_cm, at the instrument's real L}
    transmission: dict  # {species: I/I0 = exp(-tau)}
    transmission_ok: dict  # {species: True if transmission_min <= transmission <= transmission_max}
    window_max_alpha: dict  # {species: max of its alpha anywhere in the window}
    window_max_position: dict  # {species: where that max is}
    peak_is_window_max: dict  # {species: False if something else in the window is clearly stronger
    # than the selected peak -- see _WINDOW_MAX_TOLERANCE}
    peak_note: dict = field(default_factory=dict)  # {species: why it isn't the window max, or None:
    # 'edge' (wing of a line outside the window), 'separation' (a taller top inside would be too
    # close to another target), 'width' (a taller top inside whose width can't be measured here)}
    window_min_transmission: float = 1.0  # lowest TOTAL I/I0 (all species) across the window
    window_min_position: float = 0.0  # where that is, cm^-1
    score: float = 0.0  # weakest peak_alpha across species -- diagnostic only, NOT the ranking
    # key: windows are ranked by _rank_key() (species in transmission range first, then
    # distance-from-range), not by raw peak strength alone.
    # --- diagnostics, filled for the final windows only (never used to rank) ---
    interferers: list = field(default_factory=list)  # [InterfererResult], strongest first
    interference: dict = field(default_factory=dict)  # {species: sum of every OTHER species' alpha
    # (interferers AND the other targets) at its line / its own alpha there}
    main_interferer: dict = field(default_factory=dict)  # {species: biggest contributor there, or None}
    temperature_coeff: dict = field(default_factory=dict)  # {species: d(alpha)/alpha/dT, %/K}
    temperature_sweep: dict = field(default_factory=dict)  # {species: [(T, alpha), ...]}
    ratio_temperature_coeff: dict = field(default_factory=dict)  # {'a/b': %/K} for isotopologue pairs
    rank_interference: dict = field(default_factory=dict)  # {species: interference ratio used for
    # RANKING -- other targets + targets' isotopologues only; empty if rank_by_interference='none'}


@dataclass
class CombinedSearchResult:
    """Return value of find_combined_windows(): the ranked windows plus, for
    plotting, each window's OWN narrow stage-2 spectra -- every window has its
    own tiny [lo, hi] range, computed independently.
    `window_spectra[i]` is {target_label: (nu_grid, alpha)} and
    `interferer_spectra[i]` is {interferer_label: (nu_grid, alpha)}, both
    matching `windows[i]`."""
    windows: list
    window_spectra: list
    interferer_spectra: list = field(default_factory=list)
    n_rejected_saturated: int = 0  # ranked candidates dropped by the window_transmission_min filter


def _line_fwhm(m_id: int, iso_id: int, nu0: float, gamma_air: float, gamma_self: float,
               n_air: float, T: float, P: float, self_fraction: float) -> float:
    f_d = doppler_fwhm(nu0, m_id, iso_id, T)
    f_l = lorentz_fwhm(gamma_air, gamma_self, n_air, T, P, self_fraction)
    return voigt_fwhm(f_l, f_d)


def _separation(peak_positions: dict, fwhm: dict, n_separation: float) -> tuple:
    """(all pairs separated?, tightest pair (a, b, separation, required) or None)."""
    names = list(peak_positions)
    ok, worst_pair, worst_margin = True, None, float("inf")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sep = abs(peak_positions[a] - peak_positions[b])
            required = n_separation * max(fwhm[a], fwhm[b])
            if sep < required:
                ok = False
            if sep - required < worst_margin:
                worst_margin, worst_pair = sep - required, (a, b, sep, required)
    return ok, worst_pair


def _separation_of(peaks: dict, n_separation: float) -> tuple:
    """_separation() for {species: (position, top, width)}."""
    return _separation({n: v[0] for n, v in peaks.items()}, {n: v[2] for n, v in peaks.items()}, n_separation)


def _climb(alpha: np.ndarray, k: int) -> int:
    """From grid index k, walk uphill to the nearest local maximum of alpha."""
    while True:
        if k > 0 and alpha[k - 1] > alpha[k]:
            k -= 1
        elif k < len(alpha) - 1 and alpha[k + 1] > alpha[k]:
            k += 1
        else:
            return k


def _refine_top(grid: np.ndarray, alpha: np.ndarray, k: int) -> tuple:
    """(position, alpha) of the TRUE top of the bump whose highest grid point is k.

    The grid only samples the curve every wavenumber_step, so its highest point can sit
    beside the real top (a few % low for a ~0.008 cm^-1 wide line). A parabola through
    log(alpha) at the 3 grid points around k gives the top between them: exact for a
    Gaussian (Doppler) line, which is what lines are at 30 mbar, and close for a Voigt.
    Works the same for a blend of two lines: it's the top of the summed curve."""
    if k == 0 or k == len(grid) - 1 or np.any(alpha[k - 1:k + 2] <= 0):
        return float(grid[k]), float(alpha[k])
    x = grid[k - 1:k + 2] - grid[k]
    a2, a1, a0 = np.polyfit(x, np.log(alpha[k - 1:k + 2]), 2)
    if a2 >= 0:
        return float(grid[k]), float(alpha[k])
    dx = float(np.clip(-a1 / (2 * a2), x[0], x[2]))
    return float(grid[k] + dx), float(np.exp(a0 + a1 * dx + a2 * dx * dx))


def _measured_fwhm(grid: np.ndarray, alpha: np.ndarray, k: int, top: float) -> float | None:
    """Full width at half `top` of the bump at grid index k, measured on the curve itself
    (so a blend gets the width of the whole bump, not of one of its lines). None if the
    curve doesn't drop to half before the window's edge or before climbing into a taller
    neighbour -- the width isn't measurable from this window then."""
    half = top / 2

    def crossing(step):
        i = k
        while 0 <= i + step < len(alpha):
            if alpha[i + step] > top:
                return None
            if alpha[i + step] <= half:  # half-height crossed between i and i+step
                x0, x1, y0, y1 = grid[i], grid[i + step], alpha[i], alpha[i + step]
                return float(x0 + (half - y0) * (x1 - x0) / (y1 - y0))
            i += step
        return None

    left, right = crossing(-1), crossing(+1)
    return None if left is None or right is None else right - left


def _target_spectra(species: dict, grid: np.ndarray, config) -> dict:
    out = {}
    for name, d in species.items():
        t = d["target"]
        out[name] = absorption_coefficient(d["table"], d["m_id"], t.iso_id, t.mole_fraction, grid,
                                           config.T, config.P, _HAPI_INTENSITY_FLOOR, t.abundance)
    return out


def _range_penalty(tau: float, tmin: float, tmax: float) -> float:
    """0 if the optical depth tau = -ln(I/I0) = alpha*L is inside the transmission range
    [tmin, tmax], else how many DECADES of absorption it is off.

    Measured on tau (proportional to alpha), not on I/I0: being 10x too weak and 10x too
    strong then both cost exactly 1. On log(I/I0), as before, a weak absorption barely
    moves I/I0 away from 1 -- 10x too weak cost 0.02, 10x too strong 0.87 -- so too-weak
    windows were almost never penalised. Takes tau itself (CombinedWindowResult.tau), not
    the transmission: exp(-tau) rounds to exactly 1.0 for a tiny tau, which would read
    back as tau = 0. tmin = 0 / tmax = 1 mean "no strong / no weak limit"."""
    tau_hi = -log(tmin) if tmin > 0 else inf  # strongest acceptable absorption
    tau_lo = -log(tmax) if tmax < 1 else 0.0  # weakest acceptable absorption
    if tau > tau_hi:
        return log10(tau / tau_hi)
    if tau < tau_lo:
        return log10(tau_lo / max(tau, 1e-300))
    return 0.0


def _interference_term(c: CombinedWindowResult, config) -> float:
    """log10(1 + worst interference / interference_max): 0 for a clean window, ~0.3 at
    interference_max, 1 at 9x it -- same 'decades' scale as _range_penalty, so with
    interference_weight=1, interference_max costs like absorbing 2x too much/too little."""
    worst = max(c.rank_interference.values(), default=0.0)
    return log10(1 + worst / config.interference_max)


def _rank_key_for(config):
    """The ranking key for config.rank_by_interference (see SearchConfig)."""
    tmin, tmax = config.transmission_min, config.transmission_max
    mode = config.rank_by_interference
    if mode in ("none", "filter"):
        return lambda c: _rank_key(c, tmin, tmax)
    if mode == "tiebreak":
        return lambda c: (sum(c.transmission_ok.values()), -max(c.rank_interference.values(), default=0.0))
    if mode == "score":
        if config.interference_max <= 0:
            raise ValueError("rank_by_interference='score' needs interference_max > 0 (it's the scale)")
        return lambda c: -(sum(_range_penalty(t, tmin, tmax) for t in c.tau.values())
                           + config.interference_weight * _interference_term(c, config))
    raise ValueError(f"rank_by_interference must be none/filter/tiebreak/score, not {mode!r}")


def _rank_key(c: CombinedWindowResult, tmin: float, tmax: float) -> tuple:
    """Final Stage-2 ranking: how many species land in the acceptable transmission
    range comes first (a window good for 3/3 gases beats one good for 1/3, even if
    that one has a stronger raw peak); the total distance-from-range of whichever
    species are still outside it breaks ties. Raw peak strength (the old sole
    criterion) no longer drives ranking on its own."""
    n_ok = sum(c.transmission_ok.values())
    penalty = sum(_range_penalty(t, tmin, tmax) for t in c.tau.values())
    return (n_ok, -penalty)


def find_combined_windows(targets: list, config: SearchConfig,
                          background: dict | None = None) -> CombinedSearchResult:
    """Find windows where EVERY target has a detectable line, all mutually
    distinguishable -- for a single instrument setting that measures several
    gases at once in one scan (the actual "simultaneous CH4/CO2/N2O
    detection" case from the project brief: one measurement, every target
    gas visible in it).

    targets: list of TargetSpecies. Two isotopologues of one molecule are two
      targets (same name, different iso_id).
    background: {molecule: mole_fraction} of background gases to check as
      interferers (e.g. species.load_background('atm_standard.csv')); None
      skips the interference check. Minor isotopologues of the targets are
      checked either way.

    STAGE 1 (cheap, whole range):
      0. Load each target's lines (its own isotopologue only: positions,
         T-corrected intensity S(T), broadening parameters) -- no spectrum yet.
      1. proxy_intensity per line = S(T) * n_total(P,T) * mole_fraction
         (* abundance / natural_abundance if a non-natural abundance is set).
         This has units of cm^-2, NOT alpha's cm^-1 -- it deliberately skips
         the line shape (no width, no Voigt) so it's cheap to compute for
         every line. It's only used to RANK candidates against each other,
         consistently across species, never reported as a physical alpha.
      2. Candidate centers = every target's own strong-line positions.
      3. For each candidate: does EVERY species have a strong line within
         delta_sigma of it? If not, reject. Otherwise take each species'
         strongest such line (by proxy_intensity) as its representative
         (stage 2 then takes the top of the real curve around it, step 7).
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
         [center - delta_sigma/2, center + delta_sigma/2].
      7. Each species' SELECTED PEAK = the top of its real curve, found
         between grid points (_refine_top), with its width at half height
         measured on the curve (_measured_fwhm) -- not a HITRAN line
         position: for two overlapping lines the top is between them.
         Start from the bump holding its stage-1 line; switch to a taller
         top elsewhere inside the window if the separation check (redone
         with the measured widths) still passes. A max at the window's edge
         is the wing of a line outside it: never selected, only flagged
         (peak_is_window_max).
         tau = peak_alpha * L_cm; transmission = exp(-tau); flag
         transmission_ok against [transmission_min, transmission_max].
      8. Filter out survivors whose weakest peak is negligible next to the best
         one found (< 0.1%). If config.rank_by_interference isn't 'none',
         compute every candidate's interference (other targets + targets'
         isotopologues) and filter/rank on it as configured. Otherwise
         rank by _rank_key: how many species land inside
         [transmission_min, transmission_max] first, then by total
         log-distance-from-range for whichever species are still outside it.
         Sort, dedup.

    SATURATION FILTER + DIAGNOSTICS (walking the ranked list until
    max_windows windows pass):
      9. Interferers: see interference.py.
     10. Total transmission across the window (targets + interferers):
         reject the window if it drops below window_transmission_min
         anywhere (e.g. a saturated line's wing at the window edge).
     11. Temperature sensitivity (%/K per species, and per isotopologue
         ratio): see temperature.py.

    Returns a CombinedSearchResult: .windows is the ranked list of
    CombinedWindowResult (best first, up to max_windows), plus the
    per-window spectra for plotting.
    """
    # Fail fast on bad input -- these used to surface only after stage 2 (minutes in).
    labels = [t.label for t in targets]
    if len(set(labels)) != len(labels):
        raise ValueError(f"the same target is listed twice: {labels}")
    target_molecules(targets)  # isotopologues of one molecule must share its mole fraction
    rank_key = _rank_key_for(config)  # unknown rank_by_interference / bad interference_max

    delta_sigma = config.delta_sigma
    n_total = number_density(config.P, config.T)

    # ============================== STAGE 1 ==============================
    species = {}
    for t in targets:
        table = ensure_lines(t.name, config.sigma_min, config.sigma_max)
        lines_ = line_arrays(table, config.sigma_min, config.sigma_max, config.T, iso_id=t.iso_id)
        m_id = molecule_id(t.name)
        abundance_scale = 1.0 if t.abundance is None else t.abundance / natural_abundance(t.name, t.iso_id)
        # proxy_intensity: S(T) * N_species, no line shape -- see module docstring.
        proxy = lines_["sw"] * (n_total * t.mole_fraction * abundance_scale)
        strong = proxy >= config.proxy_cutoff
        # Strong-line indices sorted by position, computed once: the candidate loop below
        # then finds each window's lines with a binary search instead of re-scanning every
        # line of every species for every candidate (was most of the runtime).
        idx_strong = np.nonzero(strong)[0]
        idx_strong = idx_strong[np.argsort(lines_["nu"][idx_strong], kind="stable")]
        species[t.label] = dict(target=t, table=table, m_id=m_id, lines=lines_, strong=strong, proxy=proxy,
                                idx_strong=idx_strong, nu_strong=lines_["nu"][idx_strong])

    all_centers = np.sort(np.concatenate(
        [d["lines"]["nu"][d["strong"]] for d in species.values() if np.any(d["strong"])]
    ))

    stage1 = []
    for center in all_centers:
        lo, hi = center - delta_sigma / 2, center + delta_sigma / 2

        peak_positions, fwhm, proxy_val = {}, {}, {}
        complete = True
        for name, d in species.items():
            a = np.searchsorted(d["nu_strong"], lo, side="left")  # same [lo, hi] inclusive bounds as before
            b = np.searchsorted(d["nu_strong"], hi, side="right")
            if a >= b:
                complete = False
                break
            candidates_here = d["idx_strong"][a:b]
            k = candidates_here[int(np.argmax(d["proxy"][candidates_here]))]
            nu0 = float(d["lines"]["nu"][k])
            # Pressure-shifted center: that's where the line actually peaks (same shift
            # HAPI applies in stage 2). ~1e-4 cm^-1 at 30 mbar, up to ~1e-2 at 1 atm.
            peak_positions[name] = nu0 + float(d["lines"]["delta_air"][k]) * config.P
            proxy_val[name] = float(d["proxy"][k])
            fwhm[name] = _line_fwhm(
                d["m_id"], d["target"].iso_id, nu0, d["lines"]["gamma_air"][k],
                d["lines"]["gamma_self"][k], d["lines"]["n_air"][k],
                config.T, config.P, d["target"].mole_fraction,
            )
        if not complete:
            continue

        separation_ok, worst_pair = _separation(peak_positions, fwhm, config.n_separation)
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
        grid = window_grid(lo, hi, config.wavenumber_step, peak_positions.values())
        alphas = _target_spectra(species, grid, config)

        # Each species is represented by the TOP OF ITS CURVE, not by a HITRAN line
        # position: for an isolated line it's the same point; for two of its lines
        # overlapping (a blend), the top is between them and no listed line sits there.
        # Two options per species:
        #   - 'own':    the top of the bump holding its stage-1 line (always tried first);
        #   - 'global': the highest top of its curve anywhere inside the window, if taller.
        # A maximum at the grid's first/last point isn't a top but the wing of a line
        # outside the window -- never used as a species' point (the saturation filter
        # below handles it).
        own, best = {}, {}
        for name, alpha in alphas.items():
            start = int(np.argmin(np.abs(grid - peak_positions[name])))
            k_own = _climb(alpha, start)
            if 0 < k_own < len(grid) - 1:
                pos, top = _refine_top(grid, alpha, k_own)
                width = _measured_fwhm(grid, alpha, k_own, top) or fwhm[name]
                own[name] = (pos, top, width)
            else:  # stage-1 line is only a shoulder on a slope -- read the curve right there
                own[name] = (peak_positions[name], float(alpha[start]), fwhm[name])
            k = int(np.argmax(alpha))
            if 0 < k < len(grid) - 1 and alpha[k] > _WINDOW_MAX_TOLERANCE * own[name][1]:
                pos, top = _refine_top(grid, alpha, k)
                width = _measured_fwhm(grid, alpha, k, top)
                if width is not None:
                    best[name] = (pos, top, width)

        chosen = dict(own)
        sep_ok, worst_pair = _separation_of(chosen, config.n_separation)
        if not sep_ok:  # measured widths can be wider than stage 1's single-line estimate
            continue
        for name, alt in best.items():  # take the taller top if separation still holds
            trial = dict(chosen, **{name: alt})
            ok_t, worst_t = _separation_of(trial, config.n_separation)
            if ok_t:
                chosen, worst_pair = trial, worst_t

        peak_note = {}
        for name, alpha in alphas.items():
            k = int(np.argmax(alpha))
            if alpha[k] <= _WINDOW_MAX_TOLERANCE * chosen[name][1]:
                peak_note[name] = None
            elif k in (0, len(grid) - 1):
                peak_note[name] = "edge"
            else:
                peak_note[name] = "separation" if name in best else "width"

        peak_positions = {n: v[0] for n, v in chosen.items()}
        fwhm = {n: v[2] for n, v in chosen.items()}
        peak_alpha, window_max_alpha, window_max_position = {}, {}, {}
        window_spectrum = {}
        ok = True
        for name, alpha in alphas.items():
            if chosen[name][1] <= 0:
                ok = False
                break
            k = int(np.argmax(alpha))
            peak_alpha[name] = chosen[name][1]
            window_max_alpha[name] = float(alpha[k])
            window_max_position[name] = float(grid[k])
            window_spectrum[name] = (grid, alpha)
        if not ok:
            continue

        tau = {name: optical_depth(p, config.L_cm) for name, p in peak_alpha.items()}
        transmission = {name: exp(-t) for name, t in tau.items()}
        transmission_ok = {
            name: config.transmission_min <= tr <= config.transmission_max
            for name, tr in transmission.items()
        }

        result = CombinedWindowResult(
            center=center, lo=lo, hi=hi,
            peak_position=peak_positions, peak_alpha=peak_alpha, fwhm=fwhm,
            worst_pair=worst_pair,
            tau=tau, transmission=transmission, transmission_ok=transmission_ok,
            window_max_alpha=window_max_alpha, window_max_position=window_max_position,
            peak_is_window_max={n: peak_note[n] is None for n in peak_alpha},
            peak_note=peak_note,
            score=min(peak_alpha.values()),
        )
        candidates.append((result, window_spectrum))

    if candidates:
        best_weakest = max(c.score for c, _ in candidates)
        candidates = [(c, spec) for c, spec in candidates if c.score >= 1e-3 * best_weakest]

    if config.rank_by_interference != "none":
        # Interference for EVERY candidate, so it can be ranked on -- from the other targets
        # and the targets' isotopologues only (see SearchConfig.rank_by_interference).
        self_interferers = interferer_list(targets, {})
        for c, spec in candidates:
            grid = next(iter(spec.values()))[0]
            *_, c.rank_interference, _ = compute_interference(
                c, grid, self_interferers, config, {n: a for n, (_, a) in spec.items()})
        if config.rank_by_interference == "filter":
            candidates = [(c, spec) for c, spec in candidates
                          if max(c.rank_interference.values()) <= config.interference_max]

    candidates.sort(key=lambda cs: rank_key(cs[0]), reverse=True)

    # ================ SATURATION FILTER + DIAGNOSTICS ====================
    # Walked in rank order, lazily: interferers are only computed for candidates that could
    # still make the final list, until max_windows have passed -- computing them for every
    # stage-2 candidate would cost ~2 s each for nothing.
    interferers = interferer_list(targets, background or {})
    ensure_interferer_lines(interferers, config.sigma_min, config.sigma_max)
    selected, selected_spectra, interferer_spectra = [], [], []
    n_rejected = 0
    for c, spec in candidates:
        if any(abs(c.center - s.center) < delta_sigma for s in selected):
            continue
        grid = next(iter(spec.values()))[0]
        target_alpha = {name: alpha for name, (_, alpha) in spec.items()}
        c.interferers, spectra, c.interference, c.main_interferer = compute_interference(
            c, grid, interferers, config, target_alpha)

        total_alpha = sum(target_alpha.values()) + sum(spectra.values(), np.zeros_like(grid))
        total_tr = np.exp(-total_alpha * config.L_cm)
        k = int(np.argmin(total_tr))
        c.window_min_transmission, c.window_min_position = float(total_tr[k]), float(grid[k])
        if c.window_min_transmission < config.window_transmission_min:
            n_rejected += 1
            continue

        c.temperature_coeff, c.temperature_sweep, c.ratio_temperature_coeff = temperature_dependence(
            c, targets, config)
        selected.append(c)
        selected_spectra.append(spec)
        interferer_spectra.append({label: (grid, alpha) for label, alpha in spectra.items()})
        if len(selected) >= config.max_windows:
            break

    return CombinedSearchResult(selected, selected_spectra, interferer_spectra, n_rejected)
