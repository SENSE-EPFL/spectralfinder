"""Entry point: find spectral window(s) that detect several gases at once
(one instrument setting sees all of them -- see find_combined_windows'
docstring in hitran_tool/windows.py for how this differs from searching
each gas separately).

Edit the parameters below and run:
    uv run run.py

H2O is deliberately left out here: at 30% mole fraction it's 4-6 orders of
magnitude more concentrated than the ppm-level trace gases, which makes a
single shared path length pointless (H2O would be saturated long before the
trace gases reach a useful optical depth). Real instruments dry the sample
before measuring trace gases for exactly this reason.
"""
from hitran_tool import SearchConfig, TargetSpecies, find_combined_windows, ppm_to_fraction
from hitran_tool.report import print_combined_report, save_combined_window_details

# --- parameters -------------------------------------------------------

targets = [
    TargetSpecies(name="CH4", mole_fraction=ppm_to_fraction(1.9)),   # current ambient background
    TargetSpecies(name="N2O", mole_fraction=ppm_to_fraction(0.33)),  # current ambient background
    TargetSpecies(name="CO2", mole_fraction=ppm_to_fraction(420)),   # current ambient background
]

config = SearchConfig(
    sigma_min=2000.0,
    sigma_max=9000.0,
    L_cm=3413.4,  # PLACEHOLDER: Herriott multipass cell length from Bereiter et al. 2020 (34.134 m) -- confirm your real value with supervisors
    T=296.0,
    P=1.0,
    s_cutoff=1e-26,
    max_windows=10,
)

# --- run ----------------------------------------------------------------

if __name__ == "__main__":
    result = find_combined_windows(targets, config)
    print_combined_report(result, [t.name for t in targets])

    paths = save_combined_window_details(result, "window_details")
    print(f"\nSaved {len(paths)} per-window plots to window_details/")
