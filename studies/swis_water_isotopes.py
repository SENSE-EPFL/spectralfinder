"""Study B (Roberto, Oct 2026): SWIS, water isotopes in pure water vapour.

Roberto's numbers: ring-down 15 us -> L = c * tau = 3e10 * 15e-6 = 4.5 km, and
100% water vapour. Targets: H2(16)O, H2(18)O and HD(16)O -- one window giving
delta-18O and delta-D in a single scan. The cell holds only water, so no air
background: the interferers are water's own minor isotopologues (H2(17)O,
HD(18)O, D2O, ...) and the targets themselves. Everything not given yet (T,
scan width, detection limit, criteria, 2000-9000 cm^-1) is SubOcean's, from
run.py. The pressure isn't known either, and it decides the answer: at
SubOcean's 30 mbar the lines of pure water are pressure-broadened and the
H2(16)O wings cover every HDO and H2(18)O peak; lower, they shrink to the
Doppler width. So the search runs at 30, 10 and 5 mbar.

    uv run studies/swis_water_isotopes.py

Writes results/swis_<P>mbar/ (CSVs + one plot per window).
"""
import contextlib
import dataclasses
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run  # noqa: E402  (SubOcean settings for everything Roberto hasn't given)
from hitran_tool import TargetSpecies, find_combined_windows, percent_to_fraction  # noqa: E402
from hitran_tool.report import print_combined_report, save_combined_window_details, save_results_csv  # noqa: E402

RINGDOWN_S = 15e-6  # Roberto
L_CM = 3e10 * RINGDOWN_S  # 4.5 km
H2O = percent_to_fraction(100)  # pure water vapour (Roberto)
ISOS = [1, 2, 4]  # H2(16)O, H2(18)O, HD(16)O; add 3 for H2(17)O
P_MBAR = [30, 10, 5]  # 30 = SubOcean's

targets = [TargetSpecies("H2O", H2O, iso_id=i) for i in ISOS]
background = {}  # no air in the cell; water's other isotopologues are checked anyway

if __name__ == "__main__":  # docs/make_study_b.py imports the settings above
    for p in P_MBAR:
        config = dataclasses.replace(run.config, L_cm=L_CM, P=p / 1013.25)
        print(f"\n{'=' * 30} P = {p} mbar  (100% H2O, L = {L_CM / 1e5:.1f} km, "
              f"{config.sigma_min:.0f}-{config.sigma_max:.0f} cm^-1) {'=' * 30}")
        with contextlib.redirect_stdout(io.StringIO()):
            result = find_combined_windows(targets, config, background)
        print_combined_report(result, [t.label for t in targets])
        out = ROOT / "results" / f"swis_{p}mbar"
        save_results_csv(result, str(out))
        save_combined_window_details(result, str(out))
