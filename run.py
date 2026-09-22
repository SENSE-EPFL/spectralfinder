"""Entry point: find spectral window(s) that detect several gases at once
(one instrument setting sees all of them -- see find_combined_windows'
docstring in hitran_tool/windows.py for the two-stage algorithm and how
this differs from searching each gas separately).

Edit the parameters below and run:
    uv run run.py

This is the project's validation case, from the supervisor: CH4 (10 ppm),
N2O (1 ppm), H2O (30%), with L_cm = c * ringdown_lifetime = 3e10 cm/s *
25e-6 s = 7.5 km (an OF-CEAS cavity, not a multipass cell). Expected
answer: a window near 2.273 um (~4399.5 cm^-1). sigma_min/sigma_max are
NOT narrowed to that answer on purpose -- the point is to check the search
finds it on its own over a wide range.
"""
from hitran_tool import SearchConfig, TargetSpecies, find_combined_windows, percent_to_fraction, ppm_to_fraction
from hitran_tool.report import print_combined_report, save_combined_window_details

# --- parameters -------------------------------------------------------

targets = [
    TargetSpecies(name="CH4", mole_fraction=ppm_to_fraction(10)),
    TargetSpecies(name="N2O", mole_fraction=ppm_to_fraction(1)),
    TargetSpecies(name="H2O", mole_fraction=percent_to_fraction(30)),
]

config = SearchConfig(
    sigma_min=2000.0,
    sigma_max=9000.0,
    L_cm=7.5e5,  # 7.5 km: OF-CEAS cavity, ringdown lifetime 25 us
    delta_sigma=0.35,  # DFB laser current-scan range (cm^-1)
    n_separation=3.0,
    transmission_min=0.05,
    transmission_max=0.20,
    T=296.0,
    P=0.05,  # PLACEHOLDER: reduced pressure typical of OF-CEAS/CRDS cavities (~few tens of mbar) --
    # at 1 atm, collisional broadening alone eats most of a 0.35 cm^-1 window; confirm real value.
    s_cutoff=1e-26,
    max_windows=10,
)

# --- run ----------------------------------------------------------------

if __name__ == "__main__":
    result = find_combined_windows(targets, config)
    print_combined_report(result, [t.name for t in targets])

    paths = save_combined_window_details(result, "window_details")
    print(f"\nSaved {len(paths)} per-window plots to window_details/")
