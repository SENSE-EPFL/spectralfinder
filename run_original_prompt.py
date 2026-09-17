"""The exact validation case from the original prompt:
CH4 (10 ppm), N2O (1 ppm), H2O (30%), searched together as co-targets
(each also acting as interferer for the other two) over 2000-9000 cm^-1.
"""
import time

from hitran_tool import SearchConfig, TargetSpecies, find_windows_multi, percent_to_fraction, ppm_to_fraction
from hitran_tool.report import print_report_multi, save_window_details

targets = [
    TargetSpecies(name="CH4", mole_fraction=ppm_to_fraction(10)),
    TargetSpecies(name="N2O", mole_fraction=ppm_to_fraction(1)),
    TargetSpecies(name="H2O", mole_fraction=percent_to_fraction(30)),
]

config = SearchConfig(
    sigma_min=2000.0,
    sigma_max=9000.0,
    max_windows=10,
)

t0 = time.time()
results = find_windows_multi(targets, config)
print(f"\n=== find_windows_multi done in {time.time()-t0:.1f}s ===\n")

print_report_multi(results)

for name, result in results.items():
    save_window_details(result, name, f"window_details/{name}")
    print(f"saved detail plots for {name} -> window_details/{name}/")

print(f"\n=== total wall time: {time.time()-t0:.1f}s ===")
