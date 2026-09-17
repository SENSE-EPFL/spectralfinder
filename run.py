"""Entry point: rank candidate detection windows for a target gas.

Edit the parameters below and run:
    uv run run.py
"""
from hitran_tool import SearchConfig, TargetSpecies, find_windows, percent_to_fraction, ppm_to_fraction
from hitran_tool.report import plot_windows, print_report

# --- parameters -------------------------------------------------------

target = TargetSpecies(name="CH4", mole_fraction=ppm_to_fraction(10))

interferers = {
    "H2O": percent_to_fraction(1.5),
    "CO2": ppm_to_fraction(420),
}

config = SearchConfig(
    sigma_min=2990.0,
    sigma_max=3170.0,
    T=296.0,
    P=1.0,
    s_cutoff=1e-26,
    max_windows=10,
)

# --- run ----------------------------------------------------------------

if __name__ == "__main__":
    result = find_windows(target, interferers, config)
    print_report(result, target.name)

    fig = plot_windows(result, target.name)
    fig.savefig("run_output.png", dpi=150)
    print("\nSaved plot to run_output.png")
