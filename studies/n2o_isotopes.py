"""Study C (Oct 2026): N2O isotopes in ambient air, SubOcean measuring conditions.

Targets (from the team): the main N2O isotopologue plus the three minor ones used
in environmental studies -- 14N15N16O (alpha, 15N in the centre), 15N14N16O
(beta, 15N at the end) and 14N14N18O. Rarer and doubly substituted ones are left
out (too rare to measure). Ambient air: N2O ~0.34 ppm (today's level; the
validation case's 1 ppm is a dissolved-gas extract), the rest of the air from
atm_standard.csv as interferers (air not dried).

Conditions: the current SubOcean instrument (run.py: L = 7.5 km, 30 mbar, 40 C,
0.55 cm^-1 scan, 1% detection limit, 5% max interference). Outside its laser's
range, a different wavelength means another laser and other mirrors, so keeping
L = 7.5 km there is an assumption.

What came out (Oct 2026):
  - within the current laser's reach (7 cm^-1 around 4398.15 cm^-1) the minor
    isotopologues absorb 0.000005-0.005%, far below the 1% limit: impossible.
  - all four at once: windows near 2190-2205 cm^-1 (4.5 um) see every one, but
    in each a minor line overlaps a main-N2O or H2O line by more than 5%.
  - alpha + beta (site preference): passes, best at 2187.79 cm^-1.
  - 18O: passes, at 2195.95 cm^-1 -- 8 cm^-1 from the site-preference window,
    so not in the same 0.55 cm^-1 scan.
  - adding CH4 as a target to alpha + beta: no window passes.

    uv run studies/n2o_isotopes.py

Writes results/n2o_<case>/ (CSVs + one plot per window).
"""
import contextlib
import dataclasses
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run  # noqa: E402  (SubOcean instrument settings and background air)
from hitran_tool import TargetSpecies, find_combined_windows, ppm_to_fraction  # noqa: E402
from hitran_tool.report import print_combined_report, save_combined_window_details, save_results_csv  # noqa: E402

N2O = ppm_to_fraction(0.34)  # ambient air
CH4 = ppm_to_fraction(1.9)  # ambient air
REFERENCE, DFB_RANGE = 4398.15, 7.0  # SubOcean window and laser tuning range, cm^-1
LASER = (REFERENCE - DFB_RANGE / 2, REFERENCE + DFB_RANGE / 2)
FULL = (2000.0, 9000.0)

# isotopologue ids: 1 = 14N2 16O (main), 2 = 14N15N16O (alpha), 3 = 15N14N16O (beta), 4 = 14N2 18O
def n2o(*isos):
    return [TargetSpecies("N2O", N2O, iso_id=i) for i in isos]


CASES = {
    "subocean_laser_all": (LASER, n2o(1, 2, 3, 4)),
    "full_all": (FULL, n2o(1, 2, 3, 4)),
    "full_site_preference": (FULL, n2o(1, 2, 3)),
    "full_18O": (FULL, n2o(1, 4)),
    "full_site_preference_ch4": (FULL, n2o(1, 2, 3) + [TargetSpecies("CH4", CH4)]),
}

if __name__ == "__main__":
    for name, ((lo, hi), targets) in CASES.items():
        config = dataclasses.replace(run.config, sigma_min=lo, sigma_max=hi)
        print(f"\n{'=' * 30} {name}: {lo:g}-{hi:g} cm^-1, "
              f"{', '.join(t.label for t in targets)} {'=' * 30}")
        with contextlib.redirect_stdout(io.StringIO()):
            result = find_combined_windows(targets, config, run.background)
        print_combined_report(result, [t.label for t in targets])
        out = ROOT / "results" / f"n2o_{name}"
        save_results_csv(result, str(out))
        save_combined_window_details(result, str(out))
