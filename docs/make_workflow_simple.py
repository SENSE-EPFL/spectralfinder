"""One-line overview of the code, into docs/workflow_simple.png (the detailed method is docs/workflow.png).

    uv run docs/make_workflow_simple.py

One box per step: what it does, which module does it, and what's left after it.
Colours as in workflow.png: grey = in/out, green = search, blue = ranking, purple = checks.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "workflow_simple.png"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GREY, GREEN, BLUE, PURPLE = "#5a5a5a", "#2e7d4f", "#1f5fbf", "#7b4fa3"

STEPS = [  # (title, colour, body lines, modules, what's left after it)
    ("Entrées", GREY, ["gaz + concentrations", "instrument : L, P, T,", "balayage du laser",
                       "critères (1–20 %, 5 %)"], "run.py · app.py", ""),
    ("Étape 1 · tri rapide", GREEN, ["raies HITRAN de chaque gaz", "(intensité seule, pas de",
                                     "forme de raie)", "chaque gaz présent ?",
                                     "raies assez séparées ?"],
     "windows.py · lines.py", "≤ 200 fenêtres candidates"),
    ("Étape 2 · vrai spectre", GREEN, ["profil de Voigt sur la", "fenêtre seulement",
                                       "pic de chaque gaz", "absorption = 1 − exp(−α·L)"],
     "windows.py · spectrum.py\nbroadening.py", "le spectre de chaque candidate"),
    ("Classement", BLUE, ["1. nb de gaz dans 1–20 %", "2. puis écart à la plage"],
     "windows.py", "liste classée"),
    ("Vérification", PURPLE, ["autres gaz (air, isotopes)", "interférence ≤ 5 % → filtre",
                              "saturation, sensibilité à T", "→ affichées seulement"],
     "interference.py\ntemperature.py", "les 10 premières qui passent"),
    ("Résultat", GREY, ["10 meilleures fenêtres", "tableau console", "CSV + un graphique",
                        "par fenêtre"], "report.py", ""),
]

fig = plt.figure(figsize=(14, 3.9), facecolor="white")
fig.text(0.015, 0.93, "Workflow de l'outil", fontsize=15, fontweight="bold", color=INK, va="center")
fig.text(0.015, 0.84, "Une recherche en deux étapes : un tri rapide sur la liste des raies, puis le vrai spectre "
         "seulement sur les candidates.", fontsize=10.5, color=INK2, va="center")

x, w, gap, top, h, head = 0.015, 0.143, 0.0234, 0.75, 0.56, 0.085
for i, (title, colour, body, modules, left) in enumerate(STEPS):
    fig.patches.append(FancyBboxPatch((x, top - h), w, h, boxstyle="round,pad=0,rounding_size=0.01",
                                      transform=fig.transFigure, fc="white", ec=colour, lw=1.6))
    fig.patches.append(FancyBboxPatch((x, top - head), w, head, boxstyle="round,pad=0,rounding_size=0.01",
                                      transform=fig.transFigure, fc=colour, ec=colour, lw=1.6))
    fig.text(x + 0.008, top - head / 2, title, fontsize=10.5, fontweight="bold", color="white", va="center")
    fig.text(x + 0.008, top - head - 0.035, "\n".join(body), fontsize=9.2, color=INK, va="top", linespacing=1.55)
    fig.text(x + 0.008, top - h + 0.025, modules, fontsize=8.5, color=MUTED, va="bottom", family="monospace",
             linespacing=1.4)
    if left:
        fig.text(x + w / 2, top - h - 0.06, left, fontsize=9, color=INK2, ha="center", va="center", style="italic")
    if i < len(STEPS) - 1:
        fig.patches.append(FancyArrowPatch((x + w + 0.003, top - h / 2), (x + w + gap - 0.003, top - h / 2),
                                           transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14,
                                           color=MUTED, lw=1.5))
    x += w + gap

fig.savefig(OUT, dpi=200, facecolor="white")
print(f"saved {OUT}")
