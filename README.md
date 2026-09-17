# HITRAN spectral-window selector -- first draft

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

Edit the parameters at the top of `run.py` (target species/concentrations,
search range, and `L_cm` -- your instrument's real, fixed path length) to
change what it searches. Default: CH4 + N2O + CO2 combined over
2000-9000 cm^-1 -- prints a ranked table and saves one zoomed plot per
window under `window_details/`. HITRAN line lists are fetched from
hitran.org on first use and cached under `data/`.

One function: `find_combined_windows(targets, config)`. Only makes sense
when the targets' concentrations are within a few orders of magnitude of
each other -- mixing ppm-level trace gases with e.g. 30% H2O in one window
doesn't work (H2O would be saturated long before the trace gases reach a
useful optical depth; see `hitran_tool/windows.py` docstring). There's no
separate "background interferer" concept -- if a species matters to the
search, it goes in `targets`.

## Structure

- `hitran_tool/species.py` -- molecule name <-> HITRAN id, ppm/percent -> mole-fraction helpers.
- `hitran_tool/lines.py` -- HITRAN line-list fetch + local cache (via HAPI), and `sliced_table()`
  (keeps per-query computation cost tied to the requested range, not to how wide the cache has grown).
- `hitran_tool/broadening.py` -- Doppler / pressure (Lorentz) / Voigt FWHM (used only for the
  peak-separation check between targets, not for the spectrum curve itself -- see `windows.py`'s docstring).
- `hitran_tool/spectrum.py` -- absorption coefficient alpha(sigma) (exact HAPI Voigt profile) and
  the Beer-Lambert optical-depth helper (`optical_depth = alpha_peak * L`).
- `hitran_tool/windows.py` -- candidate window search, metrics, ranking (`find_combined_windows`).
- `hitran_tool/report.py` -- text table + per-window zoomed plots.

## Inputs (`TargetSpecies` + `SearchConfig`)

- Targets: list of `TargetSpecies(name, mole_fraction)` -- one per gas that must be simultaneously
  detectable (use `ppm_to_fraction`/`percent_to_fraction` to convert). No separate interferer list --
  every species considered has to be a target.
- `L_cm`: the instrument's real, fixed path length (cm) -- an input, not something the tool solves
  for (this is a calibration tool for an existing/known setup, not an instrument-design tool).
- Search range `sigma_min`/`sigma_max` (cm^-1), temperature `T` (K), pressure `P` (atm).
- `s_cutoff`: minimum HITRAN line intensity (cm^-1/(molecule*cm^-2)) at T, used to drop negligible lines (~1e-28 to 1e-25 is the typical range).
- `delta_sigma`: window width (cm^-1); auto-picked tight around each candidate's real peak positions if left `None`.
- `n_separation`: required gap between every PAIR of targets, in units of the larger of the two lines' Voigt FWHM -- a hard filter.
- `tau_min`/`tau_max`: acceptable optical depth range `-ln(I/I0)` at `L_cm` (default 0-40%) -- flags `CombinedWindowResult.tau_ok`
  per species, reported in the output but NOT filtered on (so a run always returns its best candidates, even when none reach this range).
- `max_windows`: how many ranked windows to return.

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

## What this first draft does and doesn't do

Implemented: line loading/caching, full-range absorption spectra per
target, envelope-based candidate windows (local maxima of the minimum
across all target spectra), pairwise target-separation check (hard filter),
tight per-candidate window sizing (built around the real peak positions
found, not a worst-case formula), and optical depth `tau = alpha_peak * L_cm`
at the instrument's real path length, flagged against an acceptable range
but not filtered on.

Deliberately deferred (per the project notes: "pas maintenant"):
isotopologue-ratio scoring (the code already threads an `iso_id` through
`TargetSpecies` so this is a natural extension), any notion of background
interferers not being searched for (kept out on purpose -- too much
complexity for now, see project discussion), T/P sensitivity for combined
windows (was implemented for an earlier single-gas search, not ported
here), and a proper sensitivity weighting for the ranking score -- the
current `score = weakest_target_peak` is a simple placeholder.

## Known limitations to revisit

- Ranking score and the n_air exponent applied to self-broadening are
  simplifications, flagged in the code, that should be reviewed once the
  detection-sensitivity model is formalized.
- No interferer/background modelling at all right now -- a real absorbing
  species that isn't in `targets` is simply invisible to the search.
