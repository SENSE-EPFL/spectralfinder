"""Study A (Roberto, Oct 2026): SubOcean CH4/N2O at high methane.

Same instrument and gases as run.py (CH4, N2O 1 ppm, H2O 30%, 30 mbar, 40 C,
L = 7.5 km), but searched only within the tuning range of the current DFB
laser: 7 cm^-1 centred on the SubOcean window (4398.15 cm^-1). For each CH4
level (10 ppm = today's case, then 1000 and 10000 ppm), is there a window in
reach of the same laser that still detects all three gases?

    uv run studies/subocean_high_ch4.py

Writes results/subocean_ch4_<ppm>ppm/ (CSVs + one plot per window).
"""
import contextlib
import dataclasses
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run  # noqa: E402  (same instrument settings and background as the validation case)
from hitran_tool import TargetSpecies, find_combined_windows, ppm_to_fraction  # noqa: E402
from hitran_tool.report import print_combined_report, save_combined_window_details, save_results_csv  # noqa: E402

REFERENCE = 4398.15  # SubOcean window, cm^-1
DFB_RANGE = 7.0  # cm^-1, typical tuning range of a DFB laser around its design wavelength
CH4_PPM = [10, 1000, 10000]

config = dataclasses.replace(run.config, sigma_min=REFERENCE - DFB_RANGE / 2,
                             sigma_max=REFERENCE + DFB_RANGE / 2, max_windows=10)


def targets_for(ppm):
    return [TargetSpecies("CH4", ppm_to_fraction(ppm))] + [t for t in run.targets if t.name != "CH4"]


if __name__ == "__main__":  # docs/make_study_figures.py imports the settings above
    for ppm in CH4_PPM:
        targets = targets_for(ppm)
        print(f"\n{'=' * 30} CH4 = {ppm} ppm  ({config.sigma_min:.2f}-{config.sigma_max:.2f} cm^-1) {'=' * 30}")
        with contextlib.redirect_stdout(io.StringIO()):
            result = find_combined_windows(targets, config, run.background)
        print_combined_report(result, [t.label for t in targets])
        out = ROOT / "results" / f"subocean_ch4_{ppm}ppm"
        save_results_csv(result, str(out))
        save_combined_window_details(result, str(out))
        ref = [i for i, c in enumerate(result.windows, 1) if c.lo <= REFERENCE <= c.hi]
        print(f"\nSubOcean window {REFERENCE}: " + (f"rank #{ref[0]}" if ref else "not in this top 10"))
