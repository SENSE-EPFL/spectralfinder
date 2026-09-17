"""Demo of find_windows_multi(): searching windows for several target gases
at once, e.g. simultaneous CH4/N2O/CO2 detection as mentioned in the project
brief.

Here: CH4 (10 ppm) and H2O (1.5%) as two co-targets over 2990-3170 cm^-1
(CH4's nu3 band), CO2 as a shared background. This band is deliberately
CH4-dense, so it also shows the point of the wrapper: H2O gets ZERO windows
under the hard separation filter here -- correct behaviour, not a bug, since
every H2O line in this band sits too close to one of CH4's ~12000 lines.
H2O's actually useful bands are elsewhere (e.g. 3400-3700 cm^-1, see
test_data.py); this demo trades scientific realism for a fast, already-cached
range so it runs quickly.
"""
from hitran_tool import SearchConfig, TargetSpecies, find_windows_multi, percent_to_fraction, ppm_to_fraction
from hitran_tool.report import print_report_multi

targets = [
    TargetSpecies(name="CH4", mole_fraction=ppm_to_fraction(10)),
    TargetSpecies(name="H2O", mole_fraction=percent_to_fraction(1.5)),
]

background = {"CO2": ppm_to_fraction(420)}

config = SearchConfig(
    sigma_min=2990.0,
    sigma_max=3170.0,
    max_windows=5,
)

results = find_windows_multi(targets, config, background)
print_report_multi(results)
