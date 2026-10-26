"""Figures for study C (N2O isotopes in ambient air), one PNG per point of the summary,
into docs/n2o/ -- meant to be attached to an email, so French and self-explanatory.

    uv run docs/make_n2o_figures.py

1_laser_actuel.png      how much each form absorbs at most with the current SubOcean
                        laser vs in a window found near 4.57 um, against the 1% limit
2_site_preference.png   the window for 15N in the centre + 15N at the end
3_oxygene_18.png        the window for 18O
4_quatre_formes.png     the best window for all four at once, and why it fails (overlap)

Each window is searched again here on a narrow range around the centre found by
studies/n2o_isotopes.py, with the same settings.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))

import n2o_isotopes as study  # noqa: E402
import run  # noqa: E402
from hitran_tool import find_combined_windows  # noqa: E402
from hitran_tool.lines import ensure_lines  # noqa: E402
from hitran_tool.species import molecule_id  # noqa: E402
from hitran_tool.spectrum import absorption_coefficient  # noqa: E402

OUT = Path(__file__).resolve().parent / "n2o"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
# one colour per form, the same in every figure (dataviz reference palette, light, slots 1-4).
# Slot 4 (yellow) is too close to orange for some colour-blind readers when they overlap,
# so 18O also gets its own dash pattern and every peak is labelled directly.
FORMS = {
    1: ("N₂O principal", "#2a78d6", "-"),
    2: ("azote 15 au centre (α)", "#eb6834", "-"),
    3: ("azote 15 au bout (β)", "#1baf7a", "-"),
    4: ("oxygène 18", "#eda100", (0, (6, 2, 1.5, 2))),
}
NICE = {"H2O": "vapeur d'eau", "CO2": "CO₂", "CO": "CO", "CH4": "CH₄", "(13C)(16O)2": "¹³CO₂",
        "HD(16O)": "vapeur d'eau (HDO)", "H2(18O)": "vapeur d'eau (H₂¹⁸O)", "(16O)(12C)(18O)": "CO₂ (¹⁸O)"}
L = run.config.L_cm

plt.rcParams.update({
    "font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.dpi": 200,
})


def fr(x, fmt=".1f"):
    return format(x, fmt).replace(".", ",")


def absorb(alpha):
    return 100 * (1 - np.exp(-np.asarray(alpha) * L))


def label_of(iso):
    return study.n2o(iso)[0].label


def window_near(center, isos):
    """The window the study found at `center`, searched again on a narrow range."""
    targets = study.n2o(*isos)
    config = dataclasses.replace(run.config, sigma_min=center - 0.6, sigma_max=center + 0.6)
    with contextlib.redirect_stdout(io.StringIO()):
        r = find_combined_windows(targets, config, run.background)
    i = min(range(len(r.windows)), key=lambda k: abs(r.windows[k].center - center))
    return r.windows[i], r.window_spectra[i], r.interferer_spectra[i]


def title(fig, text, sub):
    fig.text(0.02, 0.965, text, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.905, sub, fontsize=10.5, color=INK2, va="top")


# ------------------------------------------------- 1. current laser vs 4.57 um ----

def max_absorption_current_laser():
    """Highest absorption of each form anywhere within the current laser's reach."""
    lo, hi = study.LASER
    grid = np.arange(lo, hi, 0.001)
    table = ensure_lines("N2O", lo, hi)
    out = {}
    for iso in FORMS:
        alpha = absorption_coefficient(table, molecule_id("N2O"), iso, study.N2O, grid,
                                       run.config.T, run.config.P, s_cutoff=0.0)
        out[iso] = absorb(alpha).max()
    return out


def figure_laser(window_all):
    c = window_all[0]
    laser = max_absorption_current_laser()
    found = {iso: 100 * (1 - c.transmission[label_of(iso)]) for iso in FORMS}

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True,
                             gridspec_kw=dict(hspace=0.55, top=0.78, bottom=0.12, left=0.25, right=0.96))
    title(fig, "Le laser actuel ne voit pas les formes secondaires du N₂O",
          "Absorption maximale de chaque forme du N₂O (air ambiant, 0,34 ppm), conditions SubOcean")
    panels = [(axes[0], laser, f"Laser actuel de SubOcean (vers {fr(1e4 / study.REFERENCE, '.2f')} µm)"),
              (axes[1], found, f"Fenêtre trouvée vers {fr(1e4 / c.center, '.2f')} µm")]
    for ax, values, name in panels:
        ax.set_title(name, loc="left", fontsize=11, color=INK, pad=8)
        ax.axvspan(1e-7, 1, color="#f0efec", zorder=0)
        ax.axvline(1, color=INK2, lw=1.2, ls="--")
        for k, iso in enumerate(FORMS):
            v = max(values[iso], 1e-7)
            ax.plot([v], [k], "o", ms=9, color=FORMS[iso][1], mec="white", mew=1.5, zorder=3)
            ax.text(v * 1.6, k, f"{fr(v, '.2g') if v < 1 else fr(v)} %", va="center", fontsize=9.5, color=INK)
        ax.set_yticks(range(len(FORMS)), [FORMS[i][0] for i in FORMS])
        ax.set_ylim(len(FORMS) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_xscale("log")
        ax.set_xlim(1e-6, 100)
    axes[0].text(1.25, -0.9, "seuil de détection : 1 %", fontsize=9, color=INK2, va="bottom")
    axes[0].text(1.5e-6, -0.9, "zone non détectable", fontsize=9, color=MUTED, va="bottom")
    axes[1].set_xlabel("absorption maximale (%, échelle logarithmique)")
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x:g}".replace(".", ",") if x >= 1e-3 else f"10$^{{{int(np.log10(x))}}}$"))
    fig.savefig(OUT / "1_laser_actuel.png")
    plt.close(fig)


# ------------------------------------------------------- 2-4. window spectra ----

def figure_window(window, isos, filename, heading, sub, note=None):
    c, spectra, others = window
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(top=0.8, bottom=0.21, left=0.09, right=0.97)
    title(fig, heading, sub)

    shown = [r for r in c.interferers if r.label in others and r.label not in spectra][:3]
    for r in shown:
        grid, alpha = others[r.label]
        ax.plot(grid, absorb(alpha), color=MUTED, lw=1.3, ls=":", label=NICE.get(r.label, r.label))
    top = 0
    for iso in isos:
        name, color, ls = FORMS[iso]
        grid, alpha = spectra[label_of(iso)]
        a = absorb(alpha)
        top = max(top, a.max())
        ax.plot(grid, a, color=color, lw=2, ls=ls, label=name)
    for iso in isos:
        name, color, _ = FORMS[iso]
        n = label_of(iso)
        x, y = c.peak_position[n], 100 * (1 - c.transmission[n])
        ax.plot([x], [y], "o", ms=11, mfc="none", mec=color, mew=2, zorder=4)
        ax.annotate(f"{name.split(' (')[0] if iso > 1 else 'principal'}\n{fr(y)} %", (x, y),
                    xytext=(0, 12), textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=INK, zorder=5)
    if note:
        n, text = note
        x, y = c.peak_position[n], 100 * (1 - c.transmission[n])
        ax.annotate(text, (x, y), xytext=(40, 70), textcoords="offset points", fontsize=9.5,
                    color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=1))

    ax.set_xlim(c.lo, c.hi)
    ax.set_ylim(0, top * 1.35)
    ax.set_xlabel(f"nombre d'onde (cm⁻¹) — balayage du laser de {fr(c.hi - c.lo, '.2f')} cm⁻¹")
    ax.set_ylabel("absorption (%)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: fr(x, ".2f")))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=9.5)
    fig.savefig(OUT / filename)
    plt.close(fig)


def describe(c, isos):
    worst = max(isos, key=lambda i: c.interference.get(label_of(i), 0))
    return 100 * c.interference[label_of(worst)]


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    w_all = window_near(2188.935, [1, 2, 3, 4])
    w_sp = window_near(2187.794, [1, 2, 3])
    w_18 = window_near(2195.951, [1, 4])

    figure_laser(w_all)
    c = w_sp[0]
    figure_window(w_sp, [1, 2, 3], "2_site_preference.png",
                  f"Azote 15 au centre et au bout : une bonne fenêtre vers {fr(1e4 / c.center, '.3f')} µm",
                  f"Les trois formes absorbent assez pour être mesurées, et les autres gaz les gênent "
                  f"peu (au plus {fr(describe(c, [1, 2, 3]))} %)")
    c = w_18[0]
    figure_window(w_18, [1, 4], "3_oxygene_18.png",
                  f"Oxygène 18 : une fenêtre vers {fr(1e4 / c.center, '.3f')} µm",
                  f"À {fr(abs(c.center - w_sp[0].center), '.0f')} cm⁻¹ de la fenêtre de l'azote 15 : "
                  f"pas dans le même balayage du laser")
    c = w_all[0]
    o18 = label_of(4)
    figure_window(w_all, [1, 2, 3, 4], "4_quatre_formes.png",
                  "Les quatre formes à la fois : une raie est toujours recouverte",
                  f"Meilleur compromis, vers {fr(1e4 / c.center, '.3f')} µm : les quatre formes absorbent "
                  f"assez, mais l'oxygène 18 est collé à une raie",
                  note=(o18, f"oxygène 18 recouvert à {fr(100 * c.interference[o18], '.0f')} %\n"
                             f"(limite fixée : 5 %)"))
    for p in sorted(OUT.glob("*.png")):
        print(p.relative_to(ROOT))
