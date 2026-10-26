"""Draw the window-selection workflow, with explanatory sketches, as
docs/workflow.png and docs/workflow.pdf.

    uv run docs/make_workflow.py

Written for a non-programmer reader (physics only, no code names). Text lives in
BLOCKS below; keep it in sync with hitran_tool/windows.py. The sketches A-G are
illustrative drawings (made-up line shapes), not computed from HITRAN.
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.textpath import TextPath

OUT = Path(__file__).resolve().parent

TAGS = {  # tag -> (label, color)
    "filtre": ("FILTRE", "#c22b2b"),
    "classement": ("CLASSEMENT", "#1f5fbf"),
    "affiche": ("AFFICHÉ", "#6b6b6b"),
}
GAS = {"CH4": "#1f77b4", "N2O": "#ff7f0e", "H2O": "#2ca02c"}
GREY, RED, GOOD, WINDOW = "#7f7f7f", "#c22b2b", "#1a7d3a", "#2e7d4f"
SMALL = 7.5  # font size inside the sketches
TITLE = dict(fontsize=8.8, fontweight="bold", loc="left", pad=5)


# ------------------------------------------------------------- sketches -----

def gauss(x, x0, h, fwhm):
    return h * np.exp(-4 * np.log(2) * ((x - x0) / fwhm) ** 2)


def lorentz(x, x0, h, fwhm):
    return h / (1 + ((x - x0) / (fwhm / 2)) ** 2)


def _frame(ax, xlabel="nombre d'onde →", ylabel="absorption"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=SMALL, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=SMALL, labelpad=3)


def sketch_line_in_window(ax):
    ax.set_title("A · une raie de chaque gaz ?", **TITLE)
    sticks = {  # the line list: one stick per HITRAN line (position, intensity)
        "CH4": ([1.2, 3.0, 6.1, 8.4], [0.55, 0.9, 0.5, 0.7]),
        "N2O": ([2.4, 5.3, 9.0], [0.5, 0.7, 0.45]),
        "H2O": ([1.8, 3.6, 7.0, 9.9], [0.8, 0.6, 0.85, 0.5]),
    }
    for lo, hi, good, note in [(2.2, 3.8, True, "✓ gardée"), (7.6, 9.2, False, "✗ pas de H2O")]:
        color = GOOD if good else RED
        ax.axvspan(lo, hi, color=color, alpha=0.13, lw=0)
        ax.text((lo + hi) / 2, 1.0, note, ha="center", va="bottom", fontsize=SMALL + 0.5,
                color=color, fontweight="bold")
    for gas, (xs, hs) in sticks.items():
        ax.vlines(xs, 0, hs, color=GAS[gas], lw=2.6, label=gas)
    ax.set_xlim(0.5, 10.4)
    ax.set_ylim(0, 1.55)
    ax.legend(loc="upper center", ncol=3, fontsize=SMALL, frameon=False, handlelength=1.0,
              columnspacing=1.4, borderaxespad=0.1)
    _frame(ax, ylabel="intensité")


def sketch_separation(ax):
    ax.set_title("B · raies assez séparées ?", **TITLE)
    for (xa, xb, good, note), (lo, hi) in zip(
            [(1.3, 3.6, True, "≥ 3 largeurs ✓"), (6.6, 7.5, False, "< 3 largeurs ✗")],
            [(0.0, 4.9), (5.1, 10.0)]):
        x = np.linspace(lo, hi, 500)
        for x0, gas in [(xa, "CH4"), (xb, "N2O")]:
            y = gauss(x, x0, 1.0, 0.7)
            ax.plot(x, y, color=GAS[gas], lw=1.8)
            ax.fill_between(x, y, color=GAS[gas], alpha=0.15, lw=0)
        ax.annotate("", (xa, 1.1), (xb, 1.1), arrowprops=dict(arrowstyle="<->", lw=1.1, color="k"))
        ax.text((lo + hi) / 2, 1.27, note, ha="center", fontsize=SMALL + 0.5,
                color=GOOD if good else RED, fontweight="bold")
    ax.axvline(5.0, color="0.85", lw=1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.5)
    _frame(ax)


def sketch_peak(ax):
    ax.set_title("C · pic retenu = vrai sommet de la courbe", **TITLE)
    x = np.linspace(0.3, 10.0, 2000)
    lo, hi = 1.6, 8.0
    a, b = gauss(x, 4.0, 0.8, 0.9), gauss(x, 4.6, 0.7, 0.9)  # two lines of the same gas, blended
    total = a + b + lorentz(x, 8.8, 12.0, 0.6)  # + a strong line just outside the window
    inside = (x >= lo) & (x <= hi)

    ax.axvspan(lo, hi, color=WINDOW, alpha=0.08, lw=0)
    ax.text(lo + 0.1, 2.32, "fenêtre balayée par le laser", ha="left", va="top", fontsize=SMALL,
            color=WINDOW, fontweight="bold")
    ax.plot(x[inside], total[inside], color=GAS["CH4"], lw=2.2)
    ax.plot(x[x >= hi], total[x >= hi], color=GAS["CH4"], lw=1.3, ls="--")
    ax.plot(x[x <= lo], total[x <= lo], color=GAS["CH4"], lw=1.3, ls="--")
    ax.vlines([4.0, 4.6], 0, [0.8, 0.7], color=GREY, lw=1.3, ls=":")

    k = int(np.argmax(np.where((x > 3) & (x < 5.5), total, 0)))
    ax.plot(x[k], total[k], "o", ms=11, mfc="none", mec=GOOD, mew=2.2)
    ax.annotate("✓ pic retenu : le sommet de la courbe\n(ici entre 2 raies, en pointillé)",
                xy=(x[k] - 0.05, total[k] + 0.08), xytext=(0.45, 1.55), fontsize=SMALL, color=GOOD,
                fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.1, relpos=(1, 0)))

    edge = float(np.interp(hi, x, total))
    ax.plot(hi, edge, "x", ms=10, color=RED, mew=2.6)
    ax.annotate("✗ plus haut au bord :\nl'aile d'une raie hors\nfenêtre → jamais pris",
                xy=(hi - 0.06, edge + 0.04), xytext=(4.95, 1.95), fontsize=SMALL, color=RED,
                fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1, relpos=(1, 0)))
    ax.text(9.12, 0.12, "raie forte\nhors fenêtre", fontsize=SMALL - 0.5, color=GREY, va="bottom")
    ax.set_xlim(0.3, 10.0)
    ax.set_ylim(0, 2.4)
    _frame(ax)


def sketch_ranking(ax):
    """Same reading as the other sketches: absorption up, wavenumber across. Each gas's
    peak top must land in the horizontal band of the target range."""
    ax.set_title("D · ni trop faible, ni trop fort : on vise 1 à 20 % d'absorption", **TITLE)
    x = np.linspace(0, 10, 1500)
    floor = 0.03  # % -- just a visual baseline on the log axis
    ax.set_yscale("log")
    ax.set_xlim(0, 10)
    ax.set_ylim(floor * 0.8, 160)
    ax.axhspan(1, 20, color=GOOD, alpha=0.13, lw=0)
    ax.axhline(floor, color="0.75", lw=1)
    ax.text(0.15, 4.5, "plage visée", fontsize=SMALL + 0.5, color=GOOD, fontweight="bold", va="center")
    ax.text(0.15, 60, "trop fort", fontsize=SMALL, color=GREY, va="center")
    ax.text(0.15, 0.12, "trop faible", fontsize=SMALL, color=GREY, va="center")
    for gas, x0, top in [("CH4", 3.0, 8.0), ("N2O", 5.6, 0.3), ("H2O", 8.2, 45.0)]:
        near = np.abs(x - x0) < 1.7
        ax.plot(x[near], floor + gauss(x[near], x0, top, 0.55), color=GAS[gas], lw=2.2)
        ax.plot(x0, top + floor, "o", ms=8, mfc="none", mec=GAS[gas], mew=2)
    # CH4: top inside the band
    ax.annotate("CH4", (3.0, 8), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom",
                fontsize=SMALL + 0.5, color=GAS["CH4"], fontweight="bold")
    ax.text(3.3, 8, "✓ dans la plage", fontsize=SMALL, color=GOOD, va="center")
    # N2O: top below the band -> gap up to 1 %
    ax.annotate("", (5.6, 1.0), (5.6, 0.36), arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.text(5.85, 0.33, "N2O", fontsize=SMALL + 0.5, color=GAS["N2O"], fontweight="bold", va="center")
    ax.text(5.45, 0.62, "écart", fontsize=SMALL - 0.5, color=RED, ha="right", va="center")
    # H2O: top above the band -> gap down to 20 %
    ax.annotate("H2O", (8.2, 45), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom",
                fontsize=SMALL + 0.5, color=GAS["H2O"], fontweight="bold")
    ax.annotate("", (8.2, 20), (8.2, 40), arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.text(8.75, 30, "écart", fontsize=SMALL - 0.5, color=RED, va="center")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([0.1, 1, 20, 100])
    ax.set_yticklabels(["0,1 %", "1 %", "20 %", "100 %"], fontsize=SMALL)
    ax.minorticks_off()
    ax.set_xlabel("nombre d'onde →", fontsize=SMALL, labelpad=3)
    ax.set_ylabel("absorption au pic", fontsize=SMALL, labelpad=3)


def sketch_saturation(ax):
    ax.set_title("F · saturation : trop de lumière absorbée", **TITLE)
    x = np.linspace(0, 10, 1000)
    tau = (gauss(x, 2.4, 0.12, 0.45) + gauss(x, 4.9, 0.09, 0.45) + gauss(x, 6.6, 0.15, 0.45)
           + lorentz(x, 10.5, 3.5, 0.8))
    tr = np.exp(-tau)
    ax.plot(x, tr, color="k", lw=1.7)
    ax.axhline(0.5, color=RED, ls="--", lw=1.1)
    ax.text(0.15, 0.53, "seuil (optionnel)", color=RED, fontsize=SMALL, va="bottom")
    ax.fill_between(x, tr, 0.5, where=tr < 0.5, color=RED, alpha=0.35, lw=0)
    ax.text(4.6, 0.72, "pics des gaz mesurés", ha="center", fontsize=SMALL, color="0.3")
    ax.annotate("raie HDO saturée au bord\n→ signalée (filtre optionnel)", xy=(9.85, 0.35), xytext=(3.9, 0.2),
                fontsize=SMALL, color=RED, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.12)
    _frame(ax, xlabel="balayage du laser →", ylabel="transmission I/I0")
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0,5", "1"], fontsize=SMALL)


def sketch_interference(ax):
    ax.set_title("E · interférence : un autre gaz sous le pic", **TITLE)
    x = np.linspace(0, 10, 1000)
    ax.plot(x, gauss(x, 4.3, 1.0, 1.3), color=GAS["CH4"], lw=2.2, label="gaz mesuré")
    ax.plot(x, lorentz(x, 7.0, 0.5, 1.5), color=GREY, lw=1.6, ls="--", label="autre gaz")
    other = float(lorentz(4.3, 7.0, 0.5, 1.5))
    ax.vlines(4.3, 0, 1.0, color="0.6", ls=":", lw=1)
    ax.plot(4.3, 1.0, "o", ms=10, mfc="none", mec=GAS["CH4"], mew=2)
    ax.plot(4.3, other, "o", ms=6, color=GREY)
    ax.annotate(f"au pic, l'autre gaz absorbe {100 * other:.0f} %\nde ce qu'absorbe le gaz mesuré\n"
                f"→ interférence = {100 * other:.0f} %\n(> 5 % : fenêtre éliminée)",
                xy=(4.35, other + 0.02), xytext=(4.85, 1.02), fontsize=SMALL, va="center",
                arrowprops=dict(arrowstyle="->", color="0.3", lw=1))
    ax.legend(loc="upper left", fontsize=SMALL, frameon=False, handlelength=1.6, borderaxespad=0.1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.35)
    _frame(ax)


def sketch_temperature(ax):
    ax.set_title("G · sensibilité à la température (effet exagéré)", **TITLE)
    x = np.linspace(0, 10, 1000)
    ax.plot(x, gauss(x, 4.2, 1.0, 1.6), color=GAS["H2O"], lw=2.2, label="à T")
    ax.plot(x, gauss(x, 4.2, 0.8, 1.75), color=GAS["H2O"], lw=1.6, ls="--", label="à T + 1 K")
    ax.annotate("", (4.2, 0.8), (4.2, 1.0), arrowprops=dict(arrowstyle="<->", lw=1.1))
    ax.text(4.45, 0.9, "Δα", fontsize=SMALL + 1.5, va="center", fontweight="bold")
    ax.text(6.3, 0.55, "coefficient =\nΔα / α par degré\n(en %/K, à pression\nconstante)",
            fontsize=SMALL, va="center")
    ax.legend(loc="upper right", fontsize=SMALL, frameon=False, handlelength=1.6, borderaxespad=0.1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.3)
    _frame(ax)


# ------------------------------------------------------------- blocks -------

# (title, subtitle, header color, [(line, tag or None, indent)], [sketches], where the sketches go)
BLOCKS = [
    ("DONNÉES DE DÉPART", "", "#5a5a5a", [
        ("Gaz à mesurer et leurs concentrations (ex. CH4 10 ppm, N2O 1 ppm, H2O 30 %)", None, 0),
        ("Instrument : longueur effective 7,5 km, 30 mbar, 40 °C, balayage du laser 0,55 cm⁻¹", None, 0),
        ("Autres gaz présents : concentrations atmosphériques standard (42 molécules)", None, 0),
        ("Paramètres de toutes les raies : base de données HITRAN", None, 0),
    ], [], None),
    ("ÉTAPE 1", "présélection rapide à partir de la liste des raies (2000–9000 cm⁻¹)", "#2e7d4f", [
        ("1. Pour chaque gaz, on garde les raies assez intenses", None, 0),
        ("(intensité × concentration)", None, 1),
        ("2. Chaque raie intense définit une fenêtre candidate", None, 0),
        ("de 0,55 cm⁻¹ autour d'elle", None, 1),
        ("3. Chaque gaz a-t-il une raie intense dans la fenêtre ?  → A", "filtre", 0),
        ("4. Raies des gaz séparées d'au moins 3 × leur largeur ?  → B", "filtre", 0),
        ("5. On garde les 200 fenêtres les plus prometteuses", None, 0),
    ], [sketch_line_in_window, sketch_separation], "right"),
    ("ÉTAPE 2", "calcul du vrai spectre d'absorption (profil de Voigt : Doppler + pression)", "#2e7d4f", [
        ("6. Spectre de chaque gaz sur la fenêtre,", None, 0),
        ("ailes des raies voisines comprises", None, 1),
        ("7. Pic retenu = un sommet du spectre de chaque gaz  → C", None, 0),
        ("celui dont l'absorption convient le mieux (plage visée) ;", None, 1),
        ("l'aile d'une raie hors de la fenêtre n'est jamais prise", None, 1),
        ("8. Séparation revérifiée avec les vraies largeurs", "filtre", 0),
        ("9. Transmission de chaque gaz à son pic : I/I0 = exp(−α · L)", None, 0),
    ], [sketch_peak], "right"),
    ("CLASSEMENT", "", "#1f5fbf", [
        ("① Nombre de gaz qui absorbent entre 1 et 20 %  → D", "classement", 0),
        ("(1 % = limite de détection : SNR requis / SNR du spectre)", None, 1),
        ("② À égalité : plus petit écart total à cette plage", "classement", 0),
        ("(10× trop faible compte autant que 10× trop fort)", None, 1),
    ], [sketch_ranking], "right"),
    ("VÉRIFICATION ET ANALYSE", "pour les meilleures fenêtres, jusqu'à en avoir 10", "#7a4fa3", [
        ("10. Spectres des autres gaz : gaz atmosphériques, isotopes des gaz mesurés, autres gaz mesurés",
         None, 0),
        ("11. Interférence au pic de chaque gaz mesuré ≤ 5 % (seuil réglable) ?  → E", "filtre", 0),
        ("12. Transmission totale minimale sur le balayage (saturation)  → F", "affiche", 0),
        ("13. Sensibilité à la température du pic (en %/K) et du rapport isotopique  → G", "affiche", 0),
    ], [sketch_interference, sketch_saturation, sketch_temperature], "below"),
    ("RÉSULTAT", "", "#5a5a5a", [
        ("Les 10 meilleures fenêtres, avec pour chaque gaz : position du pic, absorption, "
         "interférence (score) et sensibilité à la température", None, 0),
        ("Un graphique du spectre par fenêtre, avec le pic retenu de chaque gaz entouré", None, 0),
    ], [], None),
]

# ------------------------------------------------------------- layout -------

W = 16.0  # figure width, inches
X0, X1 = 0.35, W - 0.35  # block left/right edges
TEXT_X = X0 + 0.25
TAG_W, TAG_X = 1.15, 7.45  # tag column (right end of the text column)
SKETCH_RIGHT_X0 = 9.0  # sketches placed to the right of the text start here
LINE_H, HEAD_H, PAD, GAP = 0.30, 0.42, 0.16, 0.42
SKETCH_H = 2.45  # one row of sketches: title + axes + axis labels
SKETCH_GAP = 0.55  # between two sketches of a row
TOP, BOTTOM = 1.0, 1.55  # page title / legend


def _text_h(lines):
    return PAD * 2 + LINE_H * len(lines)


def _height(lines, sketches, where):
    if not sketches:
        return HEAD_H + _text_h(lines)
    if where == "right":
        return HEAD_H + max(_text_h(lines), SKETCH_H + 2 * PAD)
    return HEAD_H + _text_h(lines) + SKETCH_H + PAD


def main():
    """Draw the full page (docs/workflow.png + .pdf). The sketch_* functions above can
    also be imported on their own (docs/slides/make_figures.py does)."""
    heights = [_height(lines, sk, where) for *_, lines, sk, where in BLOCKS]
    H = TOP + sum(heights) + GAP * (len(BLOCKS) - 1) + BOTTOM

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")


    def _sketch_row(sketches, x0, x1, top):
        """One row of sketches in [x0, x1] (inches), hanging from y=top."""
        n = len(sketches)
        width = (x1 - x0 - SKETCH_GAP * (n - 1)) / n
        for i, draw in enumerate(sketches):
            left = x0 + i * (width + SKETCH_GAP) + 0.3  # room for the y label
            bottom = top - SKETCH_H + 0.45  # room for the x label
            sub = fig.add_axes([left / W, bottom / H, (width - 0.3) / W, (SKETCH_H - 0.45 - 0.38) / H])
            draw(sub)


    def _width_in(text, size, bold=False):
        path = TextPath((0, 0), text, size=size, prop=FontProperties(weight="bold" if bold else "normal"))
        return path.get_extents().width / 72


    ax.text(W / 2, H - TOP / 2, "Sélection des fenêtres spectrales — principe de la méthode",
            ha="center", va="center", fontsize=17, fontweight="bold")

    y = H - TOP
    for (title, subtitle, color, lines, sketches, where), h in zip(BLOCKS, heights):
        top = y
        ax.add_patch(FancyBboxPatch((X0, top - h), X1 - X0, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="white", ec=color, lw=1.8))
        ax.add_patch(FancyBboxPatch((X0, top - HEAD_H), X1 - X0, HEAD_H,
                                    boxstyle="round,pad=0,rounding_size=0.08", fc=color, ec=color, lw=1.8))
        ax.text(X0 + 0.18, top - HEAD_H / 2, title, color="white", fontsize=12, fontweight="bold", va="center")
        if subtitle:
            ax.text(X0 + 0.18 + _width_in(title, 12, bold=True) + 0.3, top - HEAD_H / 2, "·  " + subtitle,
                    color="white", fontsize=10, va="center")

        ly = top - HEAD_H - PAD - LINE_H / 2
        for text, tag, indent in lines:
            ax.text(TEXT_X + 0.3 * indent, ly, text, fontsize=10, va="center",
                    color="#222222" if indent == 0 else "#555555")
            if tag:
                label, tcolor = TAGS[tag]
                ax.add_patch(FancyBboxPatch((TAG_X, ly - 0.105), TAG_W, 0.21,
                                            boxstyle="round,pad=0,rounding_size=0.05", fc=tcolor, ec="none"))
                ax.text(TAG_X + TAG_W / 2, ly, label, color="white", fontsize=8, fontweight="bold",
                        ha="center", va="center")
            ly -= LINE_H

        if where == "right":
            ax.plot([SKETCH_RIGHT_X0 - 0.3] * 2, [top - HEAD_H - 0.15, top - h + 0.15], color="0.85", lw=1)
            _sketch_row(sketches, SKETCH_RIGHT_X0, X1 - 0.2, top - HEAD_H - PAD)
        elif where == "below":
            _sketch_row(sketches, TEXT_X - 0.1, X1 - 0.2, top - HEAD_H - _text_h(lines))

        y = top - h - GAP
        if title != BLOCKS[-1][0]:
            ax.add_patch(FancyArrowPatch((W / 2, top - h - 0.04), (W / 2, y + 0.04),
                                         arrowstyle="-|>", mutation_scale=18, color="#555555", lw=1.6))

    # legend, one item per row, + a note on the sketches
    ly = BOTTOM - 0.45
    for tag, desc in [("filtre", "la fenêtre est éliminée si la condition n'est pas remplie"),
                      ("classement", "décide de l'ordre des fenêtres"),
                      ("affiche", "reporté seulement, n'influence pas le choix")]:
        label, tcolor = TAGS[tag]
        ax.add_patch(FancyBboxPatch((X0 + 0.1, ly - 0.1), 1.05, 0.2, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=tcolor, ec="none"))
        ax.text(X0 + 0.625, ly, label, color="white", fontsize=7.5, fontweight="bold", ha="center", va="center")
        ax.text(X0 + 1.3, ly, desc, fontsize=9, va="center", color="#333333")
        ly -= 0.27
    ax.text(X1, BOTTOM - 0.45, "Schémas A–G : illustrations du raisonnement, pas à l'échelle.",
            ha="right", va="center", fontsize=9, color="#555555", style="italic")

    fig.savefig(OUT / "workflow.png", dpi=170)
    fig.savefig(OUT / "workflow.pdf")
    print(f"saved {OUT / 'workflow.png'} and {OUT / 'workflow.pdf'}")


if __name__ == "__main__":
    main()
