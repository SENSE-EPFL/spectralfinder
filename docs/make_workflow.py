"""Draw the window-selection method for the report, as two figures:

    uv run docs/make_workflow.py

- docs/workflow.pdf / .png: the procedure step by step (inputs, stage 1, stage 2, ranking,
  checks, output), each step tagged FILTER / RANK / REPORTED;
- docs/workflow_criteria.pdf / .png: the sketches A-I those steps point to.

Sized for an A4 report: drawn 6.3 in (16 cm) wide, the text width with 2.5 cm margins, so
included at \\textwidth the text comes out at 7-8 pt. Written for a reader who knows the
physics, not the code: keep the text in sync with hitran_tool/windows.py. The sketches are
illustrations (made-up line shapes), not computed from HITRAN; docs/slides/make_figures.py
draws them too.
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.textpath import TextPath

matplotlib.rcParams["pdf.fonttype"] = 42  # real fonts in the PDF (selectable text), not Type 3 outlines

OUT = Path(__file__).resolve().parent

INK, INK2, MUTED, GRID = "#1f1f1e", "#52514e", "#898781", "#d9d8d1"
GOOD, RED = "#1a7d3a", "#c22b2b"  # ✓ / ✗ marks only; the text next to them stays in ink
BAND = "#ecebe6"  # neutral shading: a window, the absorption range
# Same hues, in the same order, as the web app's plots (app.py SERIES_LIGHT): measured gases first.
GAS = {"CH4": "#2a78d6", "N2O": "#eb6834", "H2O": "#1baf7a"}
LABEL = {"CH4": "CH₄", "N2O": "N₂O", "H2O": "H₂O"}
OTHER = MUTED  # any other species
TAGS = {  # tag -> (label, colour)
    "filter": ("FILTER", "#c22b2b"),
    "rank": ("RANK", "#1f5fbf"),
    "reported": ("REPORTED", "#6b6b6b"),
}
SMALL = 6.6  # font size inside the sketches (docs/slides/make_figures.py changes it)


# ------------------------------------------------------------- helpers ------

def _width_pt(text, size, bold=False):
    """Width of `text` in points, in matplotlib's default font."""
    path = TextPath((0, 0), text, size=size, prop=FontProperties(weight="bold" if bold else "normal"))
    return path.get_extents().width


def _title(ax, text):
    ax.set_title(text, fontsize=SMALL + 1, fontweight="bold", loc="left", pad=4, color=INK)


def _status(ax, xy, ok, text, ha="left", va="center"):
    """A green ✓ or a red ✗, then `text` in ink, at data point xy; ha="center" centres the pair."""
    icon = "✓" if ok else "✗"
    gap = _width_pt(icon, SMALL, bold=True) + 0.35 * SMALL
    width = gap + max(_width_pt(line, SMALL) for line in text.split("\n"))
    shift = -width / 2 if ha == "center" else 0
    common = dict(textcoords="offset points", ha="left", va=va, fontsize=SMALL)
    ax.annotate(icon, xy, xytext=(shift, 0), color=GOOD if ok else RED, fontweight="bold", **common)
    ax.annotate(text, xy, xytext=(shift + gap, 0), color=INK, **common)


def _arrow(ax, start, end, colour=INK2):
    ax.annotate("", end, start, arrowprops=dict(arrowstyle="->", color=colour, lw=0.8, shrinkA=1, shrinkB=2))


def _frame(ax, xlabel="wavenumber →", ylabel="absorption"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color(INK2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=SMALL, labelpad=2, color=INK2)
    ax.set_ylabel(ylabel, fontsize=SMALL, labelpad=2, color=INK2)


def _log_ticks(ax, axis, ticks, labels):
    getattr(ax, f"set_{axis}ticks")(ticks)
    getattr(ax, f"set_{axis}ticklabels")(labels, fontsize=SMALL - 0.4, color=INK2)
    ax.tick_params(axis=axis, length=2, width=0.6, pad=1.5, color=INK2)
    ax.minorticks_off()


def gauss(x, x0, h, fwhm):
    return h * np.exp(-4 * np.log(2) * ((x - x0) / fwhm) ** 2)


def lorentz(x, x0, h, fwhm):
    return h / (1 + ((x - x0) / (fwhm / 2)) ** 2)


# ------------------------------------------------------------- sketches -----

def sketch_line_in_window(ax):
    _title(ax, "A · A line of every gas?")
    sticks = {  # the line list: one stick per HITRAN line (position, intensity)
        "CH4": ([1.2, 3.0, 6.1, 8.4], [0.55, 0.9, 0.5, 0.7]),
        "N2O": ([2.4, 5.3, 9.0], [0.5, 0.7, 0.45]),
        "H2O": ([1.8, 3.6, 7.0, 9.9], [0.8, 0.6, 0.85, 0.5]),
    }
    for lo, hi, ok, note in [(2.2, 3.8, True, "kept"), (7.6, 9.2, False, "no H₂O")]:
        ax.axvspan(lo, hi, color=BAND, lw=0)
        _status(ax, ((lo + hi) / 2, 1.0), ok, note, ha="center", va="bottom")
    for gas, (xs, hs) in sticks.items():
        ax.vlines(xs, 0, hs, color=GAS[gas], lw=1.8, label=LABEL[gas])
    ax.set_xlim(0.5, 10.4)
    ax.set_ylim(0, 1.6)
    ax.legend(loc="upper center", ncol=3, fontsize=SMALL, frameon=False, handlelength=0.9, handletextpad=0.4,
              columnspacing=1.0, borderaxespad=0.0, labelcolor=INK)
    _frame(ax, ylabel="line intensity")


def sketch_separation(ax):
    _title(ax, "B · Lines far enough apart?")
    for (xa, xb, ok, note), (lo, hi) in zip(
            [(1.3, 3.6, True, "≥ 3 FWHM"), (6.6, 7.5, False, "< 3 FWHM")], [(0.0, 4.9), (5.1, 10.0)]):
        x = np.linspace(lo, hi, 500)
        for x0, gas in [(xa, "CH4"), (xb, "N2O")]:
            y = gauss(x, x0, 1.0, 0.7)
            ax.plot(x, y, color=GAS[gas], lw=1.4)
            ax.fill_between(x, y, color=GAS[gas], alpha=0.15, lw=0)
        ax.annotate("", (xa, 1.12), (xb, 1.12),
                    arrowprops=dict(arrowstyle="<->", lw=0.8, color=INK, shrinkA=0, shrinkB=0))
        _status(ax, ((lo + hi) / 2, 1.25), ok, note, ha="center", va="bottom")
    ax.axvline(5.0, color=GRID, lw=0.8)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.6)
    _frame(ax)


def sketch_peak(ax):
    _title(ax, "C · Peak: top of the curve")
    x = np.linspace(0.3, 10.0, 2000)
    lo, hi = 1.6, 8.0
    a, b = gauss(x, 4.0, 0.8, 0.9), gauss(x, 4.6, 0.7, 0.9)  # two lines of the same gas, blended
    total = a + b + lorentz(x, 8.8, 12.0, 0.6)  # + a strong line just outside the window
    inside = (x >= lo) & (x <= hi)

    ax.axvspan(lo, hi, color=BAND, lw=0)
    ax.text(lo + 0.15, 2.33, "scanned window", ha="left", va="top", fontsize=SMALL, color=INK2)
    ax.plot(x[inside], total[inside], color=GAS["CH4"], lw=1.5)
    ax.plot(x[x >= hi], total[x >= hi], color=GAS["CH4"], lw=1.0, ls="--")
    ax.plot(x[x <= lo], total[x <= lo], color=GAS["CH4"], lw=1.0, ls="--")
    ax.vlines([4.0, 4.6], 0, [0.8, 0.7], color=MUTED, lw=0.9, ls=":")  # the two blended lines

    k = int(np.argmax(np.where((x > 3) & (x < 5.5), total, 0)))
    ax.plot(x[k], total[k], "o", ms=7, mfc="none", mec=INK, mew=1.2)
    _arrow(ax, (2.4, 1.75), (x[k] - 0.12, total[k] + 0.1))
    _status(ax, (0.55, 1.92), True, "top of the curve")

    # a maximum at the window's edge: the wing of the line outside, never used
    edge = float(np.interp(hi, x, total))
    ax.plot(hi, edge, "x", ms=6, color=RED, mew=1.6, clip_on=False)
    ax.text(hi - 0.3, edge + 0.03, "edge", fontsize=SMALL, color=INK, ha="right", va="center")
    ax.set_xlim(0.3, 10.0)
    ax.set_ylim(0, 2.4)
    _frame(ax)


def sketch_ranking(ax):
    """Absorption up (log), wavenumber across: each gas's peak top must land in the band."""
    _title(ax, "D · Absorption in the range?")
    x = np.linspace(0, 10, 1500)
    floor = 0.03  # % -- just a visual baseline on the log axis
    ax.set_yscale("log")
    ax.set_xlim(0, 10)
    ax.set_ylim(floor * 0.8, 200)
    ax.axhspan(1, 20, color=BAND, lw=0)
    ax.text(0.2, 4.5, "range", fontsize=SMALL, color=INK, va="center", fontweight="bold")
    ax.text(0.2, 60, "too strong", fontsize=SMALL, color=INK2, va="center")
    ax.text(0.2, 0.15, "too weak", fontsize=SMALL, color=INK2, va="center")
    for gas, x0, top in [("CH4", 4.0, 8.0), ("N2O", 6.3, 0.3), ("H2O", 8.6, 50.0)]:
        near = np.abs(x - x0) < 1.2
        ax.plot(x[near], floor + gauss(x[near], x0, top, 0.45), color=GAS[gas], lw=1.5)
        ax.plot(x0, top + floor, "o", ms=5, mfc="none", mec=GAS[gas], mew=1.2)
    ax.annotate("CH₄", (4.0, 8), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                fontsize=SMALL, color=INK)
    ax.annotate("N₂O", (6.3, 0.3), xytext=(-5, 0), textcoords="offset points", ha="right", va="center",
                fontsize=SMALL, color=INK)
    ax.annotate("H₂O", (8.6, 50), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                fontsize=SMALL, color=INK)
    for x0, start, end in [(6.3, 0.36, 1.0), (8.6, 44, 20)]:  # distance to the range
        ax.annotate("", (x0, end), (x0, start), arrowprops=dict(arrowstyle="->", color=RED, lw=1.0,
                                                                  shrinkA=0, shrinkB=0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color(INK2)
    ax.set_xticks([])
    _log_ticks(ax, "y", [0.1, 1, 20, 100], ["0.1 %", "1 %", "20 %", "100 %"])
    ax.set_xlabel("wavenumber →", fontsize=SMALL, labelpad=2, color=INK2)
    ax.set_ylabel("absorption at the peak", fontsize=SMALL, labelpad=2, color=INK2)


def sketch_interference(ax):
    _title(ax, "E · Interference at a peak")
    x = np.linspace(0, 10, 1000)
    ax.plot(x, gauss(x, 3.6, 1.0, 1.3), color=GAS["CH4"], lw=1.5, label="measured gas")
    ax.plot(x, lorentz(x, 6.6, 0.5, 1.5), color=OTHER, lw=1.2, ls="--", label="other species")
    other = float(lorentz(3.6, 6.6, 0.5, 1.5))
    ax.vlines(3.6, 0, 1.0, color=MUTED, ls=":", lw=0.8)
    ax.plot(3.6, 1.0, "o", ms=6, mfc="none", mec=INK, mew=1.1)
    ax.plot(3.6, other, "o", ms=3.5, color=INK2)
    _arrow(ax, (5.0, 0.55), (3.7, other + 0.02))
    ax.text(5.0, 0.8, f"at the peak: other /\nmeasured = {100 * other:.0f} %\n(limit 5 %)", fontsize=SMALL,
            color=INK, va="center")
    ax.legend(loc="upper left", fontsize=SMALL, frameon=False, handlelength=1.5, borderaxespad=0.0,
              labelcolor=INK)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.45)
    _frame(ax)


def sketch_saturation(ax):
    _title(ax, "F · Saturation")
    x = np.linspace(0, 10, 1000)
    tau = (gauss(x, 2.4, 0.12, 0.45) + gauss(x, 4.9, 0.09, 0.45) + gauss(x, 6.6, 0.15, 0.45)
           + lorentz(x, 10.5, 3.5, 0.8))
    tr = np.exp(-tau)
    ax.plot(x, tr, color=INK, lw=1.3)
    ax.axhline(0.5, color=MUTED, ls="--", lw=0.8)
    ax.text(0.15, 0.53, "optional threshold", color=INK2, fontsize=SMALL - 0.4, va="bottom")
    ax.text(4.6, 0.72, "peaks of the gases", ha="center", fontsize=SMALL - 0.4, color=INK2)
    k = int(np.argmin(tr))
    ax.plot(x[k], tr[k], "o", ms=4, color=INK, clip_on=False)
    _arrow(ax, (6.1, 0.25), (x[k] - 0.15, tr[k] + 0.01))
    ax.text(2.0, 0.25, "lowest transmission:\nreported", fontsize=SMALL, color=INK, va="center")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.12)
    _frame(ax, xlabel="laser scan →", ylabel="total transmission I/I₀")
    _log_ticks(ax, "y", [0, 0.5, 1], ["0", "0.5", "1"])


def sketch_temperature(ax):
    _title(ax, "G · Temperature sensitivity")
    x = np.linspace(0, 10, 1000)
    ax.plot(x, gauss(x, 3.6, 1.0, 1.6), color=GAS["H2O"], lw=1.5, label="at T")
    ax.plot(x, gauss(x, 3.6, 0.8, 1.75), color=GAS["H2O"], lw=1.1, ls="--", label="at T + 1 K")
    ax.annotate("", (3.6, 0.8), (3.6, 1.0), arrowprops=dict(arrowstyle="<->", lw=0.8, color=INK,
                                                          shrinkA=0, shrinkB=0))
    ax.text(3.85, 0.9, "Δα", fontsize=SMALL + 0.6, va="center", color=INK)
    ax.text(5.9, 0.5, "coefficient\n= Δα / α per K\n(constant P;\nexaggerated)", fontsize=SMALL, va="center",
            color=INK)
    ax.legend(loc="upper right", fontsize=SMALL, frameon=False, handlelength=1.5, borderaxespad=0.0,
              labelcolor=INK)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.3)
    _frame(ax)


def sketch_extra_peaks(ax):
    """Same axes as D: the selected peak and another top of the same gas, both in the range."""
    _title(ax, "H · Other peaks in the range")
    x = np.linspace(0, 10, 1500)
    floor = 0.03
    tops = [(3.0, 9.0), (5.8, 2.5), (8.6, 0.3)]
    curve = floor + sum(gauss(x, x0, top, 0.45) for x0, top in tops)
    ax.set_yscale("log")
    ax.set_xlim(0, 10)
    ax.set_ylim(floor * 0.8, 200)
    ax.axhspan(1, 20, color=BAND, lw=0)
    ax.text(0.2, 4.5, "range", fontsize=SMALL, color=INK, va="center", fontweight="bold")
    ax.plot(x, curve, color=GAS["CH4"], lw=1.5)
    (x1, t1), (x2, t2), (x3, t3) = tops
    ax.plot(x1, t1 + floor, "o", ms=6, mfc="none", mec=INK, mew=1.2)
    ax.plot(x2, t2 + floor, "o", ms=6, mfc="none", mec=INK2, mew=1.0)
    ax.annotate("selected", (x1, t1), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                fontsize=SMALL, color=INK)
    _status(ax, (x2 + 0.45, t2), True, "also in range:\nlisted")
    ax.annotate("too weak", (x3, t3), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                fontsize=SMALL - 0.4, color=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color(INK2)
    ax.set_xticks([])
    _log_ticks(ax, "y", [0.1, 1, 20, 100], ["0.1 %", "1 %", "20 %", "100 %"])
    ax.set_xlabel("wavenumber →", fontsize=SMALL, labelpad=2, color=INK2)
    ax.set_ylabel("absorption", fontsize=SMALL, labelpad=2, color=INK2)


def sketch_background_limit(ax):
    """Interference at a peak vs. one background gas's concentration: everything else (rest)
    plus this gas, proportional to its concentration -- the limit is where it reaches 5 %."""
    _title(ax, "I · Background-gas limit")
    rest, own = 0.2, 0.3  # % of interference: the other species; this gas at its standard level
    f = np.logspace(-1, 2.3, 300)  # concentration / standard
    limit = (5 - rest) / own
    ax.loglog(f, rest + own * f, color=INK, lw=1.3)
    ax.axhline(5, color=RED, ls="--", lw=0.9)
    ax.text(0.12, 6.2, "limit (5 %)", fontsize=SMALL - 0.4, color=INK2, va="bottom")
    ax.plot(1, rest + own, "o", ms=4, color=INK)
    ax.annotate("standard", (1, rest + own), xytext=(4, -3), textcoords="offset points",
                ha="left", va="top", fontsize=SMALL, color=INK)
    ax.plot(limit, 5, "o", ms=5, mfc="white", mec=RED, mew=1.2)
    ax.vlines(limit, 0.08, 5, color=MUTED, ls=":", lw=0.8)
    ax.annotate(f"{limit:.0f} ×\nstandard", (limit, 5), xytext=(4, -5), textcoords="offset points",
                ha="left", va="top", fontsize=SMALL, color=INK)
    ax.set_xlim(0.1, 200)
    ax.set_ylim(0.08, 20)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color(INK2)
    _log_ticks(ax, "x", [0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    _log_ticks(ax, "y", [0.1, 1, 10], ["0.1 %", "1 %", "10 %"])
    ax.set_xlabel("background gas (× standard)", fontsize=SMALL, labelpad=2, color=INK2)
    ax.set_ylabel("interference at the peak", fontsize=SMALL, labelpad=2, color=INK2)


SKETCHES = [sketch_line_in_window, sketch_separation, sketch_peak, sketch_ranking, sketch_interference,
            sketch_saturation, sketch_temperature, sketch_extra_peaks, sketch_background_limit]


# ------------------------------------------------------------- procedure ----

# (title, subtitle, header colour, [(line, tag or None, indent)]); letters A-I = SKETCHES
BLOCKS = [
    ("INPUTS", "example: the SubOcean case, the web app's defaults", "#5a5a5a", [
        ("Gases to measure and their concentrations: CH₄ 10 ppm, N₂O 1 ppm, H₂O 30 %", None, 0),
        ("an isotopologue is a gas of its own, e.g. ¹³CH₄ or ¹⁴N¹⁵N¹⁶O", None, 1),
        ("Instrument: effective path length L = 7.5 km, P = 30 mbar, T = 40 °C, laser scan Δσ = 0.55 cm⁻¹",
         None, 0),
        ("Search range (cm⁻¹ or µm), absorption range 1–20 %, maximum interference 5 %", None, 0),
        ("Background gases: standard atmosphere (42 molecules, editable). Line parameters: HITRAN", None, 0),
    ]),
    ("STAGE 1", "fast pre-selection on the line list, no line shape", "#2e7d4f", [
        ("1.  Keep each gas's strong lines: intensity S(T) × its number density ≥ threshold", None, 0),
        ("2.  Each strong line is the centre of a candidate window of width Δσ", None, 0),
        ("3.  Does every gas have a strong line in the window?  → A", "filter", 0),
        ("4.  For each gas, take the line whose estimated absorption best fits the range", None, 0),
        ("5.  Is every pair of gases at least 3 line widths (FWHM) apart?  → B", "filter", 0),
        ("6.  Rank on that estimate, drop overlapping windows, keep the best 200", None, 0),
    ]),
    ("STAGE 2", "exact spectrum on each candidate window (Voigt profile: Doppler + pressure)", "#2e7d4f", [
        ("7.  Spectrum of each gas over the window, wings of the neighbouring lines included", None, 0),
        ("8.  Each gas's peak: a top of its curve, the one that best fits the absorption range  → C", None, 0),
        ("a maximum at the window's edge is the wing of a line outside it: never selected", None, 1),
        ("9.  Separation checked again, with the widths measured on the curves", "filter", 0),
        ("10. Absorption at each peak: A = 1 − exp(−α L)", None, 0),
    ]),
    ("RANKING", "", "#1f5fbf", [
        ("①  Number of gases whose peak absorption is in the range 1–20 %  → D", "rank", 0),
        ("1 % = detection limit: required SNR / SNR of the spectrum = 10 / 1000", None, 1),
        ("②  For equal numbers: smallest total distance to the range, in decades", "rank", 0),
        ("10 × too weak counts as much as 10 × too strong", None, 1),
    ]),
    ("CHECKS AND DIAGNOSTICS", "down the ranked list, until 10 windows pass", "#7a4fa3", [
        ("11. Spectra of all other species: background gases, other isotopologues, the other gases", None, 0),
        ("12. Interference at each gas's peak ≤ 5 %?  → E", "filter", 0),
        ("13. Lowest total transmission across the scan (saturation)  → F", "reported", 0),
        ("14. Temperature sensitivity of each peak (%/K) and of isotope ratios  → G", "reported", 0),
        ("15. Other peaks of the measured gases that also fit the range  → H", "reported", 0),
        ("16. Highest concentration of each background gas before check 12 fails  → I", "reported", 0),
    ]),
    ("OUTPUT", "", "#5a5a5a", [
        ("The 10 best windows. For each gas: peak position, absorption, interference and main", None, 0),
        ("interferer, temperature sensitivity; a plot of each window. Saved as a file to reopen, or CSV", None, 0),
    ]),
]

W = 6.3  # inches: the text width of an A4 page with 2.5 cm margins
X0, X1 = 0.02, W - 0.02  # block edges
TEXT_X = X0 + 0.12
TAG_W, TAG_H = 0.66, 0.118
TAG_X = X1 - 0.1 - TAG_W
LINE_H, HEAD_H, PAD, GAP = 0.158, 0.235, 0.07, 0.2
FONT, SUB_FONT, HEAD_FONT, TAG_FONT = 7.6, 7.0, 8.2, 6.0
LEGEND_H = 0.3


def _tag(ax, x, y, tag, width=TAG_W):
    label, colour = TAGS[tag]
    ax.add_patch(FancyBboxPatch((x, y - TAG_H / 2), width, TAG_H, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=colour, ec="none"))
    ax.text(x + width / 2, y, label, color="white", fontsize=TAG_FONT, fontweight="bold", ha="center",
            va="center_baseline")


def _block_height(lines):
    return HEAD_H + 2 * PAD + LINE_H * len(lines)


def draw_workflow():
    heights = [_block_height(lines) for *_, lines in BLOCKS]
    H = 0.02 + sum(heights) + GAP * (len(BLOCKS) - 1) + LEGEND_H
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    top = H - 0.02
    for i, ((title, subtitle, colour, lines), h) in enumerate(zip(BLOCKS, heights)):
        ax.add_patch(FancyBboxPatch((X0, top - h), X1 - X0, h, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="white", ec=colour, lw=0.9))
        ax.add_patch(FancyBboxPatch((X0, top - HEAD_H), X1 - X0, HEAD_H,
                                    boxstyle="round,pad=0,rounding_size=0.05", fc=colour, ec=colour, lw=0.9))
        ax.text(X0 + 0.1, top - HEAD_H / 2, title, color="white", fontsize=HEAD_FONT, fontweight="bold",
                va="center_baseline")
        if subtitle:
            ax.text(X0 + 0.1 + _width_pt(title, HEAD_FONT, bold=True) / 72 + 0.12, top - HEAD_H / 2,
                    "·  " + subtitle, color="white", fontsize=FONT - 0.3, va="center_baseline")

        y = top - HEAD_H - PAD - LINE_H / 2
        for text, tag, indent in lines:
            ax.text(TEXT_X + 0.24 * indent, y, text, fontsize=SUB_FONT if indent else FONT,
                    color=INK2 if indent else INK, va="center_baseline")
            if tag:
                _tag(ax, TAG_X, y, tag)
            y -= LINE_H

        if i < len(BLOCKS) - 1:
            ax.add_patch(FancyArrowPatch((W / 2, top - h - 0.02), (W / 2, top - h - GAP + 0.02),
                                         arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1.0))
        top -= h + GAP

    # legend: what each tag means
    x, y = X0 + 0.02, LEGEND_H / 2 - 0.02
    for tag, meaning in [("filter", "the window is dropped if it fails"), ("rank", "decides the order"),
                         ("reported", "shown, does not select")]:
        _tag(ax, x, y, tag, width=0.62)
        ax.text(x + 0.62 + 0.07, y, meaning, fontsize=SUB_FONT, color=INK2, va="center_baseline")
        x += 0.62 + 0.07 + _width_pt(meaning, SUB_FONT) / 72 + 0.3
    return fig


def draw_criteria():
    fig, axes = plt.subplots(3, 3, figsize=(W, 5.5), layout="constrained")
    # rect: a small right margin, which the left-aligned titles of the last column don't get otherwise
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.05, wspace=0.05, hspace=0.07, rect=(0, 0, 0.985, 1))
    for ax, draw in zip(axes.flat, SKETCHES):
        draw(ax)
    return fig


def main():
    for name, draw in [("workflow", draw_workflow), ("workflow_criteria", draw_criteria)]:
        fig = draw()
        fig.savefig(OUT / f"{name}.pdf")
        fig.savefig(OUT / f"{name}.png", dpi=300)
        plt.close(fig)
        print(f"saved {OUT / name}.pdf and .png")


if __name__ == "__main__":
    main()
