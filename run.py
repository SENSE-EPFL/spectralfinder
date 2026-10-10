"""Entry point: find spectral window(s) that detect several gases at once
(one instrument setting sees all of them -- see find_combined_windows'
docstring in hitran_tool/windows.py for the two-stage algorithm and how
this differs from searching each gas separately).

Edit the parameters below and run:
    uv run run.py

This is the project's validation case, from the supervisor: CH4 (10 ppm),
N2O (1 ppm), H2O (30%), with L_cm = c * ringdown_lifetime = 3e10 cm/s *
25e-6 s = 7.5 km (an OF-CEAS cavity, not a multipass cell). sigma_min/
sigma_max are NOT narrowed to any expected answer on purpose -- the point
is to check the search finds it on its own over a wide range.

Two reference answers to check against: an earlier, rougher answer near
2.273 um (~4399.5 cm^-1), and the real window used on the SubOcean
CH4-N2O instrument, 4398.15 cm^-1 -- the one the search should ideally
surface now that transmission_min/max, delta_sigma and P are confirmed
real values rather than placeholders.
"""
from hitran_tool import (SearchConfig, TargetSpecies, find_combined_windows, load_background,
                         percent_to_fraction, ppm_to_fraction)
from hitran_tool.report import print_combined_report, save_combined_window_details, save_results_csv

# --- parameters -------------------------------------------------------

targets = [
    TargetSpecies(name="CH4", mole_fraction=ppm_to_fraction(10)),
    TargetSpecies(name="N2O", mole_fraction=ppm_to_fraction(1)),
    TargetSpecies(name="H2O", mole_fraction=percent_to_fraction(30)),
    # Isotopologues are separate targets: same molecule, same (total) mole fraction,
    # different iso_id -- HITRAN's natural abundance gives its share. E.g. 13CH4:
    # TargetSpecies(name="CH4", mole_fraction=ppm_to_fraction(10), iso_id=2),
]

# Background gases checked as interferers in every final window (reported, not ranked on).
# A molecule that's also a target takes the target's concentration instead.
background = load_background("atm_standard.csv")

config = SearchConfig(
    sigma_min=2000.0,
    sigma_max=9000.0,
    L_cm=7.5e5,  # 7.5 km: OF-CEAS cavity, ringdown lifetime 25 us
    delta_sigma=0.55,  # DFB scan range, cm^-1 -- confirmed 0.3-0.55 cm^-1 for OF-CEAS, using the
    # upper end (more candidate windows survive)
    n_separation=3.0,  # confirmed
    transmission_min=0.80,  # at most 20% absorption at a target's peak (I/I0 >= 0.80)
    spectrum_snr=1000.0,  # typical spectrum SNR (Roberto) -> weakest usable absorption =
    snr_min=10.0,  # snr_min / spectrum_snr = 1% (replaces transmission_max: I/I0 <= 0.99)
    window_transmission_min=0.0,  # saturation filter on the total I/I0 across the scan: off
    T=313.15,  # 40 C, SubOcean's cell temperature
    P=30 / 1013.25,  # 30 mbar, confirmed value for the SubOcean instrument
    rank_by_interference="filter",  # drop windows where a target's interference > interference_max
    interference_max=0.05,  # (other modes: 'none' | 'tiebreak' | 'score' -- see SearchConfig)
    max_windows=10,
    temperature_step_K=1.0,  # T sensitivity: alpha at T, T+1, ..., T+5 K (constant P)
    temperature_steps=5,
    temperature_stability_K=0.1,  # cell T stability -> flags isotope ratios over ratio_precision
)

# --- run ----------------------------------------------------------------

if __name__ == "__main__":
    result = find_combined_windows(targets, config, background)
    print_combined_report(result, [t.label for t in targets])

    paths = save_combined_window_details(result, "window_details")
    print(f"\nSaved {len(paths)} per-window plots to window_details/")
    csv_paths = save_results_csv(result, "results")
    print(f"Saved {', '.join(csv_paths)}")
