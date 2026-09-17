# HITRAN spectral-window selector -- first draft

Ranks candidate spectral windows for detecting a target gas (given its
concentration) against a set of interfering species, using line-by-line
HITRAN data via [HAPI](https://hitran.org/hapi/).

## Setup

```
uv sync
```

## Run the demo

```
uv run example_ch4_detection.py
```

Ranks windows for detecting 10 ppm CH4 in humid air (H2O, CO2 as
interferers) over 2990-3170 cm^-1, prints a ranked table, and saves
`example_ch4_detection.png`. HITRAN line lists are fetched from
hitran.org on first use and cached under `data/`.

## Structure

- `hitran_tool/species.py` -- molecule name <-> HITRAN id, ppm/percent -> mole-fraction helpers.
- `hitran_tool/lines.py` -- HITRAN line-list fetch + local cache (via HAPI).
- `hitran_tool/broadening.py` -- Doppler / pressure (Lorentz) / Voigt FWHM.
- `hitran_tool/spectrum.py` -- absorption coefficient (Beer-Lambert alpha) and path-length helpers.
- `hitran_tool/windows.py` -- candidate window search, metrics, ranking (`find_windows`).
- `hitran_tool/report.py` -- text table + quick-look plot.

## Inputs (`TargetSpecies` + `SearchConfig`)

- Target molecule + concentration (mole fraction; use `ppm_to_fraction`/`percent_to_fraction`).
- Interferers: `{name: mole_fraction}`, given explicitly by the caller -- no background/atmospheric composition is assumed or auto-populated (kept out on purpose to avoid adding a second, implicit concentration source).
- Search range `sigma_min`/`sigma_max` (cm^-1), temperature `T` (K), pressure `P` (atm).
- `s_cutoff`: minimum HITRAN line intensity (cm^-1/(molecule*cm^-2)) at T, used to drop negligible lines (~1e-28 to 1e-25 is the typical range).
- `delta_sigma`: window width (cm^-1); auto-picked from the local line width if left `None`.
- `n_separation`: required gap to the nearest interferer line, in units of the larger of the two lines' Voigt FWHM.
- `tau_target`/`tau_tolerance`: target optical depth `-ln(I/I0)` (default 0.10 +/- 30%), used to back out a suggested path length per window.
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

Implemented: line loading/caching, full-range absorption spectra for target +
interferers, peak-centered candidate windows, interference ratio, minimum
line-separation check (hard filter by default), suggested path length from
the target optical depth, and a cheap +/-10 K / +/-0.1 atm sensitivity
finite-difference for the selected windows.

Deliberately deferred (per the project notes: "pas maintenant"):
isotopologue-ratio scoring (the code already threads an `iso_id` through
`TargetSpecies` so this is a natural extension), and a proper
sensitivity/interference weighting for the ranking score -- the current
`score = peak_alpha / (1 + interference_index)` is a simple placeholder.

## Known limitations to revisit

- Ranking score and the n_air exponent applied to self-broadening are
  simplifications, flagged in the code, that should be reviewed once the
  detection-sensitivity model is formalized.
- Runtime: interferer + target spectra are computed once per species/window
  set, but the current line-separation search still re-scans interferer line
  lists per candidate; fine for a first draft, worth vectorizing if
  `max_windows`/search ranges grow a lot.
