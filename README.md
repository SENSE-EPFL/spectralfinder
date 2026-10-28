# SpectralFinder

Finds the spectral window(s) where several target gases (given their
concentrations) are all simultaneously detectable in ONE measurement --
i.e. picks the best setting for a single-scan, single-instrument multi-gas
sensor. This is the case the project targets: one measurement has to see
every target gas at once, not a different setting per gas. Uses
line-by-line [HITRAN](https://hitran.org) data via [HAPI](https://hitran.org/hapi/).

SpectralFinder is developed independently at EPFL; it is not developed,
maintained or endorsed by the HITRAN team.

## Citation

If you use SpectralFinder, please cite it:

> Bruttin, L., Grilli, R., Chappellaz, J., & Thomas, E. (2026). *SpectralFinder* (version 1.2.0)
> [Software]. https://github.com/SENSE-EPFL/spectralfinder

The DOI will be added here with the first release archived on Zenodo. The same
reference is in `CITATION.cff` (GitHub's "Cite this repository" button reads it).

## Data

Line data are downloaded from HITRANonline and cached locally (`data/`, see
`hitran_tool/lines.py`), with their retrieval date. HITRAN is updated
continuously, also between editions, so results can change with the data
version: report the retrieval date with them. Please also cite HITRAN and HAPI:

- I. E. Gordon, L. S. Rothman, R. J. Hargreaves et al., *The HITRAN2024 molecular
  spectroscopic database*, J. Quant. Spectrosc. Radiat. Transfer 353, 109807 (2026),
  [doi:10.1016/j.jqsrt.2026.109807](https://doi.org/10.1016/j.jqsrt.2026.109807)
- R. V. Kochanov, I. E. Gordon, L. S. Rothman, P. Wcisło, C. Hill, J. S. Wilzewski,
  *HITRAN Application Programming Interface (HAPI): A comprehensive approach to working
  with spectroscopic data*, J. Quant. Spectrosc. Radiat. Transfer 177, 15-30 (2016),
  [doi:10.1016/j.jqsrt.2016.03.005](https://doi.org/10.1016/j.jqsrt.2016.03.005)

## Setup

```
uv sync
```

## Run

```
uv run run.py
```

Or, with a graphical interface in the browser:

```
uv run app.py
```

`app.py` runs the same search with every parameter set from the page (gases and
isotopes, instrument as L or ringdown time, pressure, temperature, scan width,
search range, criteria), then shows the ranked windows and an interactive spectrum
of each one. The default case is SubOcean around its window (a few seconds).
The background gases default to `atm_standard.csv` and can be edited in the
page (concentrations changed, gases removed or added -- e.g. for an unusual
environment). A simulation can be saved (one JSON file: parameters, background
gases and every result, spectra included -- `hitran_tool/saving.py`) and
reopened in the page later without searching again; its tables also download
as a zip of the CSVs below, plus `background.csv`.

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
(`atm_standard.csv`) are checked as interferers in every final window: by
default a window is dropped if any target's interference exceeds 5%
(`interference_max`) -- if a species must be *detected*, it goes in
`targets`.

Outputs, per window: the ranked console table, a per-window detail block
(selected line of each target, T sensitivity in %/K, interference at the
line, the targets' other lines that also fit the absorption range, and the
background gases to watch -- the concentration above which each one would
break the interference limit), one plot under `window_details/`, and five
CSVs under `results/` (`windows.csv`, `interferers.csv`, `temperature.csv`,
`extra_peaks.csv`, `background_limits.csv`).

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
  Each target is represented by a **top of its real curve** in the window
  (refined between grid points, width measured at half height on the curve)
  -- not by a HITRAN line position: when two of its lines overlap, the top is
  between them. Which top: the one whose absorption best fits the target
  range (the tallest among equally good ones; `pick_lines_for_range`) -- at
  high concentration the strongest line saturates while weaker ones are
  usable. Stage 1 already picks each gas's line the same way, from an
  estimated peak absorption. A max at the window's edge (wing of a line
  outside) is never selected, only flagged. Converts that peak to
  transmission `I/I0 = exp(-alpha*L_cm)` at the instrument's real path length
  and flags whether that's in the acceptable range.
- **Interference filter** (default): walking the ranked list, a window is
  dropped if any target's interference (below) exceeds `interference_max`.
- **Saturation filter** (optional, off by default): the total transmission
  across the whole scanned window (targets + all interferers) must stay >=
  `window_transmission_min`; the lowest value is always reported.
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
    isotopologues of one molecule, the same for their ratio (= the difference
    of their two coefficients, to first order), flagged when it exceeds
    `ratio_precision` (1 permil) for the cell's `temperature_stability_K`.

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
- `hitran_tool/saving.py` -- save / reopen a whole simulation (inputs + results) as JSON.
- `atm_standard.csv` -- standard atmospheric mole fractions used as background interferers.
- `app.py` -- Dash interface (`uv run app.py`).
- `hitran_tool/worker.py` -- runs one search in a separate process (the app's process pool).
- `prefetch.py` -- fills `data/` with every molecule over 2000-9000 cm^-1 (run when the container starts).

## Deployment (public web app)

The Docker image (`Dockerfile`) holds no HITRAN data, as the HITRAN team asked: when the
container starts, `prefetch.py` downloads the line lists it doesn't have yet from HITRANonline
into `data/` -- one request per molecule, 59 in all (~410 MB) -- then the app serves. Mount a
volume on `/app/data` so that happens only on the first deployment, not on every restart. The
app itself runs with `HITRAN_OFFLINE=1`: it never downloads, so a search outside 2000-9000
cm^-1 is refused with a message. The retrieval date is stored with each line list and shown
at the bottom of the page.

```
docker compose up        # -> http://localhost:8501 (health check: /health)
```

Releases (`.github/workflows/publish-container.yml`): every push to `main` builds the image
and deploys it to the dev site (EPFL network/VPN only) within ~5 min; a version tag such as
`1.0.0` (`git tag 1.0.0 && git push origin 1.0.0`) deploys it to the public site, once approved.

- **Parallel searches.** HAPI keeps its state in globals, so concurrent searches in one
  process would overwrite each other's tables. Each search runs in its own worker process
  (`hitran_tool/worker.py`), at most `MAX_PARALLEL_SEARCHES` at once (default 2, set with
  `docker run -e`); further searches wait in line. Identical searches are served from
  the app's cache without recomputing. Sizing: ~0.5 GB RAM and one CPU core per worker,
  plus the server itself (~70 MB idle, more with many visitors and cached results); a search takes ~10 s on a 7 cm^-1 range, ~2 min on
  the full 2000-9000 cm^-1.
- **`app.py` starts the server only under `if __name__ == "__main__"`**: worker
  processes are started with `spawn`, which re-imports the main script, and must not start
  the server.
- **Dependencies** are pinned in `uv.lock`. This project sits in a uv workspace on the
  author's machine, so a plain `uv lock` here updates the workspace's lock, not this one:
  regenerate it from a copy of `pyproject.toml` + `README.md` outside the workspace.

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
- `window_transmission_min`: optional saturation filter on the total I/I0 anywhere in the window
  (default 0 = off).
- `transmission_min`/`transmission_max`: acceptable `I/I0 = exp(-alpha*L_cm)` range (default 0.80-0.95, i.e. 5-20% attenuation)
  -- flags `CombinedWindowResult.transmission_ok` per species, reported in the output but NOT
  filtered on (so a run always returns its best candidates, even when none reach this range).
- `spectrum_snr`/`snr_min`: detection limit. If `spectrum_snr` is set, the weakest acceptable
  absorption is `snr_min / spectrum_snr` (same rule for every species) and replaces
  `transmission_max` -- `run.py` uses 1000 and 10, i.e. 1%.
- `pick_lines_for_range` (default True): pick each gas's line/peak by fit to the absorption range
  rather than by strength (False = strongest line, the original trace-gas behaviour).
- `rank_by_interference` (`filter` | `none` | `tiebreak` | `score`, default `filter`),
  `interference_max` (default 5%), `interference_weight`: how interference enters the selection
  -- see `SearchConfig` in `windows.py`. `filter` uses the full reported interference (background
  CSV included); `tiebreak`/`score` need it for every candidate, so they use the other targets and
  the targets' isotopologues only.
- `temperature_stability_K` (default 0.1 K), `ratio_precision` (default 1e-3): flag for the isotope
  ratio T sensitivity.
- `max_windows`: how many ranked windows to return.
- `max_stage1_candidates`: cap on how many Stage-1 candidates get the expensive Stage-2 treatment (default 200).
- `wavenumber_step`: stage-2 grid step (default 0.002 cm^-1) -- peak tops are refined between grid
  points (checked against a 2e-5 cm^-1 grid: within 0.02%), so it mostly shapes the plotted curves.
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
not ranked on), per-window plots with each target's selected line marked,
other in-range lines of each target (with their own T sensitivity), and the
limit concentration of every background gas before the window fails the
interference limit.

Deliberately deferred: ranking on interference or T sensitivity (only the
saturation filter uses the interferers), T sensitivity of the interference
itself, P sensitivity.

## Known limitations to revisit

- `P` (cell/cavity pressure) has a first-order effect on whether anything can
  be found at all: at 1 atm, collisional broadening widens lines enough that
  3 species can't fit pairwise-separated inside a 0.35 cm^-1 window. `run.py`
  uses 30 mbar, the confirmed SubOcean value.
- Stage 1 only knows the targets: it can propose windows next to a strong
  line of a non-target (e.g. HDO at 30% H2O). The saturation filter removes
  the worst ones afterwards, but the search itself isn't steered away.
- The saturation filter also applies to the targets' own lines: a target
  absorbing more than 50% (I/I0 < 0.5) removes the window, even though the
  per-target transmission range is otherwise only flagged.
- Isotopologue pairs: the ranking prefers each species' strongest line, so
  a minor isotopologue (e.g. 13CH4, ~1%) ends up ~100x weaker than the main
  one -- nothing yet looks for pairs of lines with comparable absorption.
- Interferers are reported, not used in the search: a window badly
  overlapped by e.g. CO2 still ranks as high as a clean one.
- Lines are only included within 5 cm^-1 around a window: a window at the
  very edge of `[sigma_min, sigma_max]` misses wings from lines beyond it.

## Which change matters? (`compare_variants.py`)

`uv run compare_variants.py` reruns the validation case adding this project's
changes one at a time (and the interference-ranking options), and prints which
windows enter/leave the top 10 at each step -- see its docstring. Results in
`results/variants*.csv`.

## Validation

The project's reference case (from the supervisor): CH4 (10 ppm), N2O (1 ppm),
H2O (30%), `L_cm` = 7.5 km (`c * 25 us` ringdown lifetime, an OF-CEAS cavity),
P = 30 mbar, T = 40 C, `delta_sigma` = 0.55 cm^-1, detection limit 1%
(SNR 10 on a spectrum SNR of 1000), interference filter at 5%.

- Searching blindly over 2000-9000 cm^-1, all top-10 windows have the three
  gases in range; the real SubOcean window (4398.15 cm^-1, inside the window
  found at 4398.00) is #7.
- Searching only the tuning range of the SubOcean DFB laser (7 cm^-1 around
  4398.15), it's #1 -- see `studies/subocean_high_ch4.py`.

## Studies (`studies/`)

- `subocean_high_ch4.py`: same instrument, CH4 at 10 / 1000 / 10000 ppm within
  the DFB's 7 cm^-1. At 10 ppm the SubOcean window is #1. At 1000 ppm and
  above no window passes: CH4 itself is measurable (weaker lines, I/I0
  0.83-0.94), but CH4 and 13CH4 absorb under every N2O and H2O peak
  (interference >= ~60% in the best window, often >1000%; still >= 28% on
  H2O even without N2O as a target).
- `swis_water_isotopes.py`: SWIS, H2(16)O + H2(18)O + HDO in 100% water vapour,
  L = 4.5 km (15 us ring-down), everything else SubOcean's, at 30 / 10 / 5 mbar
  (SWIS's pressure isn't known yet). At 30 mbar no window passes: the
  pressure-broadened H2(16)O wings put HDO and H2(18)O over 5% interference.
  At 10 mbar 2 windows pass, at 5 mbar 5; the best is 4175.1 cm^-1 (all three
  at 8-15% absorption, interference <= 1.5%), but its isotope ratios move by
  2-3 permil per 0.1 K. The counts at 10 and 5 mbar depend on
  `max_stage1_candidates` (2000 instead of 200: 5 mbar -> 10 windows, 10 mbar
  -> 5, none with all three in range, the two above being replaced by
  neighbouring candidates in the stage-1 dedup); 30 mbar stays at 0. Figures:
  `uv run docs/make_study_figures.py` -> `docs/study_a.png`, `docs/study_a_ch4.png`
  (the SubOcean window at the three CH4 levels), `docs/study_b.png`,
  `docs/study_b_pressure.png` (the 4175.1 window at the three pressures).
