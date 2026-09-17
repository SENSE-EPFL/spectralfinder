"""First-draft demo: rank candidate windows for detecting CH4 at 10 ppm in air.

Search range: 2990-3170 cm^-1 (CH4 nu3 fundamental, ~3.2-3.3 um), a band
commonly used for mid-IR CH4 sensing (see Lechevallier et al. 2019). H2O and
CO2 are included as interferers since both have real absorption there.
"""
from hitran_tool import (
    SearchConfig,
    TargetSpecies,
    find_windows,
    percent_to_fraction,
    ppm_to_fraction,
)
from hitran_tool.report import plot_windows, print_report

target = TargetSpecies(name="CH4", mole_fraction=ppm_to_fraction(10))

interferers = {
    "H2O": percent_to_fraction(1.5),  # ~1.5% water vapour, typical near-surface humid air
    "CO2": ppm_to_fraction(420),  # current atmospheric background
}

config = SearchConfig(
    sigma_min=2990.0,
    sigma_max=3170.0,
    T=296.0,
    P=1.0,
    s_cutoff=1e-26,
    n_separation=5.0,
    tau_target=0.10,
    max_windows=10,
)

result = find_windows(target, interferers, config)
print_report(result, target.name)

if __name__ == "__main__":
    fig = plot_windows(result, target.name)
    fig.savefig("example_ch4_detection.png", dpi=150)
    print("\nSaved plot to example_ch4_detection.png")
