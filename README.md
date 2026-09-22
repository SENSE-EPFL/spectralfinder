# HITRAN spectral-window selector

Finds the spectral window(s) where several target gases (given their
concentrations) are all simultaneously detectable in ONE measurement --
i.e. picks the best setting for a single-scan, single-instrument multi-gas
sensor. This is the case the project targets: one measurement has to see
every target gas at once, not a different setting per gas. Uses
line-by-line HITRAN data via [HAPI](https://hitran.org/hapi/).

## Setup

```
uv sync
```

## Run

```
uv run run.py
```

Edit the parameters at the top of `run.py` to change what it searches: target
species/concentrations, search range, and the instrument's real numbers
(`L_cm` -- path length, `delta_sigma` -- laser scan range, `P` -- cell
pressure). Prints a ranked table and saves one zoomed plot per window under
`window_details/`. HITRAN line lists are fetched from hitran.org on first
use and cached under `data/`.

One function: `find_combined_windows(targets, config)`. Only makes sense
when the targets' concentrations are within a few orders of magnitude of
each other -- mixing ppm-level trace gases with e.g. 30% H2O only works if
the path length is long enough for the trace gases without saturating H2O
(see the validation case below, which does exactly this). There's no
separate "background interferer" concept -- if a species matters to the
search, it goes in `targets`.

### Two-stage search (see `hitran_tool/windows.py` docstring for detail)

Scanning a wide range with the real, full-physics spectrum (Doppler +
collisional broadening, exact Voigt profile) is slow -- HAPI's per-line loop
pays for every line in range regardless of whether that region is ever a
plausible answer. So the search runs in two passes:

- **Stage 1** (cheap, whole range): works only from line positions and a
  proxy intensity (`S(T) * n_total(P,T) * mole_fraction` -- no line shape at
  all), to find candidate windows where every target has a line and every
  pair is separated enough. No spectrum is computed in this stage.
- **Stage 2** (expensive, narrow): only for the candidates Stage 1 keeps,
  computes the real absorption spectrum -- but just over that candidate's
  own tiny window (`delta_sigma`, e.g. 0.3-0.4 cm^-1, a DFB laser's
  current-scan range), which is cheap precisely because the range is tiny.
  Converts to transmission `I/I0 = exp(-alpha*L_cm)` at the instrument's
  real path length and flags whether that's in the acceptable range.

## Structure

- `hitran_tool/species.py` -- molecule name <-> HITRAN id, ppm/percent -> mole-fraction helpers.
- `hitran_tool/lines.py` -- HITRAN line-list fetch + local cache (via HAPI), and `sliced_table()`
  (keeps per-query computation cost tied to the requested range, not to how wide the cache has grown).
- `hitran_tool/broadening.py` -- Doppler / pressure (Lorentz) / Voigt FWHM -- cheap, closed-form,
  used by Stage 1's separation check (never calls HAPI's spectrum routine).
- `hitran_tool/spectrum.py` -- absorption coefficient alpha(sigma) (exact HAPI Voigt profile),
  number density, and the Beer-Lambert optical-depth helper -- only ever called in Stage 2.
- `hitran_tool/windows.py` -- the two-stage candidate search and ranking (`find_combined_windows`).
- `hitran_tool/report.py` -- text table + per-window zoomed plots (out-of-range values flagged
  with `[brackets]` in the table, red/green text in the plots).

## Inputs (`TargetSpecies` + `SearchConfig`)

- Targets: list of `TargetSpecies(name, mole_fraction)` -- one per gas that must be simultaneously
  detectable (use `ppm_to_fraction`/`percent_to_fraction` to convert). No separate interferer list --
  every species considered has to be a target.
- `L_cm`: the instrument's real, fixed path length (cm) -- an input, not something the tool solves
  for (this is a calibration tool for an existing/known setup, not an instrument-design tool). For
  a ringdown cavity, `L_cm = c * ringdown_lifetime`.
- Search range `sigma_min`/`sigma_max` (cm^-1), temperature `T` (K), pressure `P` (atm) -- P matters
  a lot: at 1 atm, collisional broadening alone can eat most of a narrow `delta_sigma` window
  (see Known limitations).
- `s_cutoff`: minimum HITRAN line intensity (cm^-1/(molecule*cm^-2)) at T, used to drop negligible lines (~1e-28 to 1e-25 is the typical range).
- `delta_sigma`: window width (cm^-1), a real hardware constraint (e.g. a DFB laser's current-tuning
  range), default 0.35 -- NOT auto-derived.
- `n_separation`: required gap between every PAIR of targets, in units of the larger of the two lines' Voigt FWHM -- a hard filter.
- `transmission_min`/`transmission_max`: acceptable `I/I0 = exp(-alpha*L_cm)` range (default 0.05-0.20)
  -- flags `CombinedWindowResult.transmission_ok` per species, reported in the output but NOT
  filtered on (so a run always returns its best candidates, even when none reach this range).
- `max_windows`: how many ranked windows to return.
- `max_stage1_candidates`: cap on how many Stage-1 candidates get the expensive Stage-2 treatment (default 200).

## Key physics/unit points (see module docstrings for detail)

- **Doppler broadening depends on T and mass only, not on pressure.** It's a
  separate, independent mechanism from collisional (pressure) broadening; the
  two combine into a Voigt profile, not a sum. (`broadening.py`)
- **HITRAN's `gamma_air`/`gamma_self` are half-widths (HWHM)** at 296 K, 1 atm;
  FWHM = 2x that, scaled to (T, P) via HAPI's own `EnvironmentDependency_Gamma0`.
- **Absorption coefficient needs the absorber's own number density, not the
  total gas density.** HAPI's `absorptionCoefficient_Voigt(..., HITRAN_units=False)`
  scales by the *total* number density from the ideal gas law (i.e. assumes
  the absorber is 100% of the gas). For a trace species at mole fraction x,
  the result must still be multiplied by x -- this is done once, explicitly,
  in `spectrum.py`, and is exactly the unit trap a naive `alpha = N*S*sigma(nu)`
  implementation falls into for ppm-level species.
- All wavenumber-based quantities are handled directly in cm^-1 -- the
  `dnu/nu = dlambda/lambda = dsigma/sigma` identity in the original notes is a
  tautology (all three ratios are the same number), not three separate
  quantities to compute.

## What's implemented / deferred

Implemented: two-stage search (cheap proxy pass, then real Voigt spectra only
on survivors), pairwise target-separation check (hard filter, cheap analytic
FWHM), transmission at the instrument's real path length (reported, not
filtered), per-window plots with the acceptable range flagged clearly.

Deliberately deferred (per the project notes: "pas maintenant"):
isotopologue-ratio scoring (the code already threads an `iso_id` through
`TargetSpecies` so this is a natural extension), any notion of background
interferers not being actively searched for, T/P sensitivity, and a proper
weighting of the ranking score against distance from the target transmission
range -- the current `score = weakest_target_peak` (real Stage-2 peak) is a
simple placeholder that doesn't yet reward being close to
`[transmission_min, transmission_max]`.

## Known limitations to revisit

- Ranking score doesn't account for transmission fit -- a window can score
  well on raw peak strength while every species is outside the acceptable
  transmission range (see the validation case: the best-positioned window
  ranks #3, not #1, for exactly this reason).
- `P` (cell/cavity pressure) has a first-order effect on whether anything can
  be found at all: at 1 atm, collisional broadening widens lines enough that
  3 species can't fit pairwise-separated inside a 0.35 cm^-1 window. Reduced-
  pressure operation (as real OF-CEAS/CRDS instruments use) is required; the
  exact value in `run.py` is still a placeholder.
- No interferer/background modelling at all right now -- a real absorbing
  species that isn't in `targets` is simply invisible to the search.

## Validation

The project's reference case (from the supervisor): CH4 (10 ppm), N2O (1 ppm),
H2O (30%), `L_cm` = 7.5 km (`c * 25 us` ringdown lifetime, an OF-CEAS cavity).
Searching blindly over 2000-9000 cm^-1 (no hint about where to look), the tool
finds a window at **4401.09 cm^-1 (2.27216 um)** -- within 0.04% of the
supervisor's stated ~2.273 um answer. At that window, H2O's transmission is
in the acceptable range; CH4 and N2O are not yet (too weak at the current
placeholder pressure) -- see Known limitations.
