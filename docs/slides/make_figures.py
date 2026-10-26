"""Figures for the Monday deck (docs/slides/build_deck.py), into docs/slides/fig/.

    uv run docs/slides/make_figures.py

- sketch_A..G.png: the explanatory sketches of docs/make_workflow.py, re-rendered
  at slide size with bigger text and without their titles (the slide gives them).
- subocean_window.png: the real SubOcean window as the code finds it (10 ppm CH4,
  DFB tuning range), with each gas's selected peak and the main interferers.
- high_ch4.png: optical depth over the DFB range with CH4 at 10 and 1000 ppm,
  N2O and H2O -- why no window survives at high methane.
- hero.png: a dark, decorative version of the real spectrum for the title slide.
"""
import contextlib
import dataclasses
import io
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs"))

import make_workflow as mw  # noqa: E402
import run  # noqa: E402
from hitran_tool import TargetSpecies, find_combined_windows, ppm_to_fraction, percent_to_fraction  # noqa: E402
from hitran_tool.lines import ensure_lines  # noqa: E402
from hitran_tool.spectrum import absorption_coefficient  # noqa: E402
from hitran_tool.species import molecule_id  # noqa: E402

OUT = Path(__file__).resolve().parent / "fig"
OUT.mkdir(exist_ok=True)
GAS = {"CH4": "#1f77b4", "N2O": "#ff7f0e", "H2O": "#2ca02c"}
REF = 4398.15  # SubOcean window
LO, HI = REF - 3.5, REF + 3.5  # DFB tuning range (7 cm^-1)


# ------------------------------------------------------------- sketches -----

def sketches():
    # (name, function, figure size in inches = size on the slide, font size)
    jobs = [
        ("A", mw.sketch_line_in_window, (5.6, 3.0), 11),
        ("B", mw.sketch_separation, (5.6, 3.0), 11),
        ("C", mw.sketch_peak, (7.6, 4.0), 11),
        ("D", mw.sketch_ranking, (6.9, 3.7), 11),
        ("E", mw.sketch_interference, (4.3, 2.8), 8.5),
        ("F", mw.sketch_saturation, (4.3, 2.8), 8.5),
        ("G", mw.sketch_temperature, (4.3, 2.8), 8.5),
        # card-sized versions for the one-slide "selection" summary
        ("A_small", mw.sketch_line_in_window, (4.0, 2.9), 8.5),
        ("C_small", mw.sketch_peak, (4.0, 2.9), 7.5),
        ("D_small", mw.sketch_ranking, (4.0, 2.9), 8),
    ]
    for name, draw, size, font in jobs:
        mw.SMALL = font
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        ax.set_title("", loc="left")  # the sketches use left-aligned titles
        fig.tight_layout(pad=0.3)
        fig.savefig(OUT / f"sketch_{name}.png", dpi=220, transparent=True)
        plt.close(fig)


# ------------------------------------------------------------- real data ----

def subocean_window():
    config = dataclasses.replace(run.config, sigma_min=LO, sigma_max=HI, max_windows=5)
    with contextlib.redirect_stdout(io.StringIO()):
        result = find_combined_windows(run.targets, config, run.background)
    i = next(i for i, c in enumerate(result.windows) if c.lo <= REF <= c.hi)
    c, spectra, others = result.windows[i], result.window_spectra[i], result.interferer_spectra[i]

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for name, (grid, alpha) in spectra.items():
        ax.plot(grid, alpha, color=GAS[name], lw=2.2, label=name)
    for r, ls in zip([r for r in c.interferers if r.label in others][:2], ["--", ":"]):
        grid, alpha = others[r.label]
        ax.plot(grid, alpha, color="0.45", lw=1.4, ls=ls, label=f"{r.label} (interférent)")
    for name in spectra:
        pos, a = c.peak_position[name], c.peak_alpha[name]
        ax.plot(pos, a, "o", ms=11, mfc="none", mec=GAS[name], mew=2.2)
        ax.annotate(f"{name}\n{100 * (1 - c.transmission[name]):.1f} %", (pos, a), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10.5,
                    color=GAS[name], fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 6)
    ax.set_xlim(c.lo, c.hi)
    ax.ticklabel_format(useOffset=False, axis="x")
    ax.set_xlabel("nombre d'onde (cm$^{-1}$)", fontsize=11)
    ax.set_ylabel("coefficient d'absorption α (cm$^{-1}$)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, loc="lower right", frameon=True, framealpha=0.92, edgecolor="none", ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "subocean_window.png", dpi=220, transparent=True)
    plt.close(fig)
    return spectra


def _tau(name, iso, x, grid, T=313.15):
    table = ensure_lines(name, 2000, 9000)
    return absorption_coefficient(table, molecule_id(name), iso, x, grid, T, run.config.P, 1e-30) * run.config.L_cm


def high_ch4():
    grid = np.arange(LO, HI, 0.002)
    tau = {
        "CH4 1000 ppm (+ ¹³CH4)": _tau("CH4", 1, 1e-3, grid) + _tau("CH4", 2, 1e-3, grid),
        "CH4 10 ppm": _tau("CH4", 1, 1e-5, grid) + _tau("CH4", 2, 1e-5, grid),
        "N2O 1 ppm": _tau("N2O", 1, 1e-6, grid),
        "H2O 30 %": _tau("H2O", 1, 0.30, grid),
    }
    style = {
        "CH4 1000 ppm (+ ¹³CH4)": dict(color=GAS["CH4"], lw=1.5),
        "CH4 10 ppm": dict(color=GAS["CH4"], lw=1.1, ls=":", alpha=0.9),
        "N2O 1 ppm": dict(color=GAS["N2O"], lw=1.8),
        "H2O 30 %": dict(color=GAS["H2O"], lw=1.5),
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    band = ax.axhspan(-np.log(0.99), -np.log(0.80), color="#1a7d3a", alpha=0.12, lw=0)
    ax.axvspan(4397.722, 4398.272, color="0.5", alpha=0.12, lw=0)
    ax.annotate("fenêtre SubOcean", (4397.997, 30), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, color="0.3", annotation_clip=False)
    for label, t in tau.items():
        ax.plot(grid, np.maximum(t, 1e-6), label=label, **style[label])
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 30)
    ax.set_xlim(LO, HI)
    ax.ticklabel_format(useOffset=False, axis="x")
    ax.set_xlabel("nombre d'onde (cm$^{-1}$) — plage d'accord du laser DFB (7 cm$^{-1}$)", fontsize=11)
    ax.set_ylabel("épaisseur optique τ = α·L", fontsize=11)
    ax.tick_params(labelsize=10)
    handles, labels = ax.get_legend_handles_labels()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.4, rect=(0, 0.1, 1, 1))  # room for the legend under the plot
    fig.legend(handles + [band], labels + ["plage visée (1–20 %)"], fontsize=9.5, loc="lower center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handlelength=1.8)
    fig.savefig(OUT / "high_ch4.png", dpi=220, transparent=True)
    plt.close(fig)


def hero():
    grid = np.arange(4396.2, 4400.4, 0.002)
    fig = plt.figure(figsize=(6.6, 7.5), facecolor="#1E2430")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#1E2430")
    for name, iso, x in [("H2O", 1, 0.30), ("CH4", 1, 1e-5), ("N2O", 1, 1e-6)]:
        t = np.maximum(_tau(name, iso, x, grid), 1e-7)
        for lw, alpha in [(9, 0.06), (4, 0.15), (1.6, 0.95)]:  # soft glow
            ax.plot(grid, t, color=GAS[name], lw=lw, alpha=alpha, solid_capstyle="round")
    ax.set_yscale("log")
    ax.set_ylim(3e-6, 2.0)
    ax.set_xlim(grid[0], grid[-1])
    ax.axis("off")
    fig.savefig(OUT / "hero.png", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def isotopes():
    """Same reading as the other sketches (absorption up, log scale): the same CH4 line
    for 12CH4 and 13CH4 -- natural abundance 1.1% makes the minor one ~90x weaker."""
    gauss = mw.gauss
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.linspace(0, 10, 1500)
    floor = 0.004
    ax.set_yscale("log")
    ax.set_xlim(0, 10)
    ax.set_ylim(floor * 0.8, 160)
    ax.axhspan(1, 20, color="#1a7d3a", alpha=0.13, lw=0)
    ax.axhline(floor, color="0.75", lw=1)
    ax.text(0.15, 4.5, "plage visée", fontsize=11.5, color="#1a7d3a", fontweight="bold", va="center")
    for label, x0, top, color in [("¹²CH4", 3.0, 12.0, GAS["CH4"]), ("¹³CH4", 6.9, 12.0 * 0.0111, "#6FA8DC")]:
        near = np.abs(x - x0) < 1.7
        ax.plot(x[near], floor + gauss(x[near], x0, top, 0.55), color=color, lw=2.4)
        ax.plot(x0, top + floor, "o", ms=9, mfc="none", mec=color, mew=2)
        ax.annotate(label, (x0, top), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom",
                    fontsize=12, color=color, fontweight="bold")
    ax.annotate("", (6.9, 0.16), (3.55, 11), arrowprops=dict(arrowstyle="->", color="#c22b2b", lw=1.4,
                                                              connectionstyle="arc3,rad=-0.25"))
    ax.text(8.55, 3.2, "≈ 90× plus faible\n(abondance 1,1 %)", fontsize=10.5, color="#c22b2b", ha="center",
            va="center", fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([0.01, 0.1, 1, 20, 100])
    ax.set_yticklabels(["0,01 %", "0,1 %", "1 %", "20 %", "100 %"], fontsize=10)
    ax.minorticks_off()
    ax.set_xlabel("nombre d'onde →", fontsize=10.5)
    ax.set_ylabel("absorption au pic", fontsize=10.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "isotopes.png", dpi=220, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    sketches()
    print("sketches done")
    subocean_window()
    print("subocean window done")
    high_ch4()
    print("high CH4 done")
    hero()
    isotopes()
    print(f"saved figures in {OUT}")
