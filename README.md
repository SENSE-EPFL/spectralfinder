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

One function: `find_combined_windows(targets, config, background)`. Only makes sense
when the targets' concentrations are within a few orders of magnitude of
each other -- mixing ppm-level trace gases with e.g. 30% H2O only works if
the path length is long enough for the trace gases without saturating H2O
(see the validation case below, which does exactly this). Background gases
(`atm_standard.csv`) are checked as interferers in every final window, but
never used to filter or rank -- if a species must be *detected*, it goes in
`targets`.

Outputs, per window: the ranked console table, a per-window detail block
(selected line of each target, T sensitivity in %/K, interference at the
line), one plot under `window_details/`, and three CSVs under `results/`
(`windows.csv`, `interferers.csv`, `temperature.csv`).

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
  Takes alpha **at each target's selected line** (the line the window was
  picked for -- not the window's max, which can be a wing from a strong line
  just outside; that case is flagged), converts to transmission
  `I/I0 = exp(-alpha*L_cm)` at the instrument's real path length and flags
  whether that's in the acceptable range. If the real spectrum shows another
  of a species' lines in the window standing taller than the Stage-1 pick,
  that line becomes the selected one (if separation still holds).
- **Saturation filter** (a hard filter): the total transmission across the
  whole scanned window (targets + all interferers) must stay >=
  `window_transmission_min` (default 0.5) -- rejects e.g. windows sitting on
  the wing of a saturated HDO line.
- **Diagnostics** (final windows only, reported, not ranked on):
  - *Interferers* (`interference.py`): every background molecule of
    `atm_standard.csv` at its standard concentration, each isotopologue
    separately, plus the targets' own minor isotopologues at the target's
    concentration. Per target: sum of every other species' alpha at its line
    (interferers AND the other targets) / its own alpha there, and the main
    contributor.
  - *Temperature* (`temperature.py`): alpha at the selected line at T, T+1,
    ..., T+5 K (constant P, so number density drops with T too), headline
    coefficient `(alpha(T+1) - alpha(T)) / alpha(T)` in %/K; for two
    isotopologues of one molecule, the same for their ratio.

## Structure

- `hitran_tool/species.py` -- molecule name <-> HITRAN id, ppm/percent -> mole-fraction helpers.
- `hitran_tool/lines.py` -- HITRAN line-list fetch + local cache (one table per molecule, all
  isotopologues, stored as `data/{molecule}.lines.npz` -- loads in ms, unlike HAPI's own text
  tables), and `sliced_table()` (keeps per-query cost tied to the requested range and isotopologue,
  not to how wide the cache has grown).
- `hitran_tool/broadening.py` -- Doppler / pressure (Lorentz) / Voigt FWHM -- cheap, closed-form,
  used by Stage 1's separation check (never calls HAPI's spectrum routine).
- `hitran_tool/spectrum.py` -- absorption coefficient alpha(sigma) (exact HAPI Voigt profile),
  number density, and the Beer-Lambert optical-depth helper -- only ever called in Stage 2.
- `hitran_tool/windows.py` -- the two-stage candidate search and ranking (`find_combined_windows`).
- `hitran_tool/interference.py` -- background/isotopologue interferers per window.
- `hitran_tool/temperature.py` -- %/K sensitivity per window, per target and per isotopologue ratio.
- `hitran_tool/report.py` -- text table + per-window detail, CSV export, per-window zoomed plots
  (selected line of each target circled and labelled, strongest interferers overlaid).
- `atm_standard.csv` -- standard atmospheric mole fractions used as background interferers.

## Inputs (`TargetSpecies` + `SearchConfig`)

- Targets: list of `TargetSpecies(name, mole_fraction, iso_id=1, abundance=None)` -- one per gas
  that must be simultaneously detectable (use `ppm_to_fraction`/`percent_to_fraction` to convert).
  Isotopologues are separate targets: `TargetSpecies("CH4", 10e-6, iso_id=2)` is 13CH4 in 10 ppm of
  CH4 -- `mole_fraction` is always the whole molecule's, HITRAN's natural abundance gives the
  isotopologue's share (override with `abundance`, e.g. to simulate another d13C).
- `background`: `{molecule: mole_fraction}` interferers, e.g. `load_background("atm_standard.csv")`.
- `L_cm`: the instrument's real, fixed path length (cm) -- an input, not something the tool solves
  for (this is a calibration tool for an existing/known setup, not an instrument-design tool). For
  a ringdown cavity, `L_cm = c * ringdown_lifetime`.
- Search range `sigma_min`/`sigma_max` (cm^-1), temperature `T` (K), pressure `P` (atm) -- P matters
  a lot: at 1 atm, collisional broadening alone can eat most of a narrow `delta_sigma` window
  (see Known limitations).
- `proxy_cutoff`: a line counts as "strong" if its PROXY intensity (sw * n_total(P,T) *
  mole_fraction, cm^-2, default 1e-12) is >= this. Filtering on the proxy rather than raw HITRAN
  sw puts every species on the same footing regardless of concentration -- a raw-sw cutoff shared
  across a 1 ppm trace gas and 30% H2O isn't comparable (a "weak" H2O line still out-absorbs a
  "strong" N2O one). A per-species-relative cutoff doesn't work either: CH4/N2O have much
  stronger fundamental bands elsewhere in a wide search range than in a combination-band region,
  which made weaker (but still real) regions vanish entirely when compared to that species' own
  global best line.
- `delta_sigma`: window width (cm^-1), a real hardware constraint (e.g. a DFB laser's current-tuning
  range), default 0.35 -- NOT auto-derived.
- `n_separation`: required gap between every PAIR of targets, in units of the larger of the two lines' Voigt FWHM -- a hard filter.
- `window_transmission_min`: saturation filter on the total I/I0 anywhere in the window (default 0.5).
- `transmission_min`/`transmission_max`: acceptable `I/I0 = exp(-alpha*L_cm)` range (default 0.05-0.20)
  -- flags `CombinedWindowResult.transmission_ok` per species, reported in the output but NOT
  filtered on (so a run always returns its best candidates, even when none reach this range).
- `max_windows`: how many ranked windows to return.
- `max_stage1_candidates`: cap on how many Stage-1 candidates get the expensive Stage-2 treatment (default 200).
- `wavenumber_step`: stage-2 grid step (default 0.002 cm^-1) -- only shapes the plotted curves, alpha at
  each selected line is exact (line centers are added to the grid).
- `temperature_step_K`/`temperature_steps`: T sweep for the sensitivity check (default +1 K x 5).

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
- **Line wings are summed out to 5 cm^-1**, not HAPI's default 50 half-widths
  (~0.3 cm^-1 at 30 mbar): the wing of a saturated neighbour (e.g. HDO at
  30% H2O) cut there is still as strong as a ppm-level target. (`spectrum.py`)
- All wavenumber-based quantities are handled directly in cm^-1 -- the
  `dnu/nu = dlambda/lambda = dsigma/sigma` identity in the original notes is a
  tautology (all three ratios are the same number), not three separate
  quantities to compute.

## What's implemented / deferred

Implemented: two-stage search (cheap proxy pass, then real Voigt spectra only
on survivors), pairwise target-separation check (hard filter, cheap analytic
FWHM), transmission at the instrument's real path length (reported, not
filtered), isotopologues as separate targets, background + isotopologue
interference check and temperature sensitivity per window (both reported,
not ranked on), per-window plots with each target's selected line marked.

Deliberately deferred: ranking/filtering on interference or T sensitivity,
P sensitivity.

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
- Interferers are reported, not used in the search: a window badly
  overlapped by e.g. CO2 still ranks as high as a clean one.
- Lines are only included within 5 cm^-1 around a window: a window at the
  very edge of `[sigma_min, sigma_max]` misses wings from lines beyond it.

## Validation

The project's reference case (from the supervisor): CH4 (10 ppm), N2O (1 ppm),
H2O (30%), `L_cm` = 7.5 km (`c * 25 us` ringdown lifetime, an OF-CEAS cavity).
Searching blindly over 2000-9000 cm^-1 (no hint about where to look), the tool
finds a window at **4401.09 cm^-1 (2.27216 um)** -- within 0.04% of the
supervisor's stated ~2.273 um answer. At that window, H2O's transmission is
in the acceptable range; CH4 and N2O are not yet (too weak at the current
placeholder pressure) -- see Known limitations.
