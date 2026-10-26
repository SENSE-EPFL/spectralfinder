"""One-page summary figure per study, into docs/study_a.png and docs/study_b.png.

    uv run studies/subocean_high_ch4.py     # first: writes results/subocean_ch4_<ppm>ppm/
    uv run studies/swis_water_isotopes.py   # first: writes results/swis_<P>mbar/
    uv run docs/make_study_figures.py

Same layout for both. Left: the input parameters, each tagged with where it comes
from. Right: how many windows survive at each tested value (from the study's CSVs),
then one spectrum recomputed here -- for A, the whole DFB range at 10 and 1000 ppm
CH4 (why nothing survives); for B, the best window at the lowest pressure.
"""
import contextlib
import csv
import dataclasses
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))

import subocean_high_ch4 as study_a  # noqa: E402  (each study's settings)
import swis_water_isotopes as study_b  # noqa: E402
from hitran_tool import find_combined_windows  # noqa: E402
from hitran_tool.interference import compute_interference, interferer_list  # noqa: E402
from hitran_tool.lines import ensure_lines  # noqa: E402
from hitran_tool.species import molecule_id  # noqa: E402
from hitran_tool.spectrum import absorption_coefficient  # noqa: E402

OUT = Path(__file__).resolve().parent
INK, INK2, MUTED, CARD, BAND = "#0b0b0b", "#52514e", "#8a8984", "#f5f4f1", "#e6e5e0"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (dataviz reference palette, light)
NICE = {"H2O": "H₂¹⁶O", "H2(18O)": "H₂¹⁸O", "HD(16O)": "HDO", "HD(18O)": "HD¹⁸O", "H2(17O)": "H₂¹⁷O",
        "D2(16O)": "D₂O", "(13C)H4": "¹³CH4"}


def fr(x, fmt=".1f"):
    return format(x, fmt).replace(".", ",")


def pct(fraction):
    return f"{fr(100 * fraction, '.0f')} %"


def num(n):
    """French thousands: 1000, 10 000."""
    return f"{n:,}".replace(",", " ") if n >= 10000 else str(n)


def n_windows(folder):
    with open(ROOT / "results" / folder / "windows.csv") as f:
        return len({row["window"] for row in csv.DictReader(f)})


def absorb(alpha, L_cm):
    return 100 * (1 - np.exp(-alpha * L_cm))


# ------------------------------------------------------------- layout -------

def card(fig, x, y, w, h):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fc=CARD, ec="none", zorder=0))


def tag(fig, x, y, label):
    fig.text(x, y, label, fontsize=8.5, color=INK2, ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.6", fc="white", ec="#d6d5d0", lw=0.8))


def page(title, question, rows, footer, tiles_title, tiles):
    """The common frame: title, parameter card, arrow, result tiles. Returns the figure."""
    fig = plt.figure(figsize=(13, 7.4), facecolor="white")
    fig.text(0.02, 0.96, title, fontsize=17, fontweight="bold", color=INK, va="center")
    fig.text(0.02, 0.915, question, fontsize=11.5, color=INK2, va="center")

    x0, y = 0.035, 0.79
    card(fig, 0.02, 0.025, 0.29, 0.795)
    fig.text(x0, y, "Paramètres d'entrée", fontsize=13, fontweight="bold", color=INK, va="top")
    y -= 0.068
    for label, value, source in rows:
        if label is None:  # section heading
            y -= 0.008
            fig.text(x0, y, value, fontsize=11.5, fontweight="bold", color=INK, va="center")
            y -= 0.047
            continue
        fig.text(x0, y + 0.012, label, fontsize=9, color=MUTED, va="center")
        fig.text(x0, y - 0.013, value, fontsize=10.5, color=INK, va="center")
        if source:
            tag(fig, 0.245, y, source)
        y -= 0.049
    fig.text(x0, 0.06, footer, fontsize=8.5, color=MUTED, va="center", linespacing=1.5)

    fig.patches.append(FancyArrowPatch((0.318, 0.45), (0.392, 0.45), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=16, color=MUTED, lw=1.6))
    fig.text(0.355, 0.475, "recherche", fontsize=9, color=MUTED, ha="center", va="bottom")

    fig.text(0.40, 0.83, tiles_title, fontsize=12, fontweight="bold", color=INK, va="center")
    x, w, gap = 0.40, 0.178, 0.012
    for head, n, note in tiles:
        card(fig, x, 0.67, w, 0.125)
        fig.text(x + 0.013, 0.772, head, fontsize=10.5, color=INK2, va="center")
        fig.text(x + 0.013, 0.715, str(n), fontsize=26, fontweight="bold", color=INK, va="center")
        fig.text(x + 0.058, 0.715, note, fontsize=9, color=INK2, va="center", linespacing=1.35)
        x += w + gap
    return fig


def spectrum_axes(fig, plot_title, notes, xlo, xhi, ymin, xlabel):
    """The absorption plot under the tiles, styled; the caller draws the curves."""
    fig.text(0.40, 0.625, plot_title, fontsize=12, fontweight="bold", color=INK, va="center")
    fig.text(0.40, 0.055, notes, fontsize=10, color=INK2, va="center", linespacing=1.7)
    ax = fig.add_axes([0.445, 0.2, 0.525, 0.33])
    ax.axhspan(1, 20, color=BAND, lw=0, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(ymin, 600)
    ticks = [t for t in [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100] if t >= ymin]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{fr(t, 'g')} %" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(xlo, xhi)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fr(v, ".1f")))
    ax.set_xlabel(xlabel, fontsize=10, color=INK2)
    ax.set_ylabel("absorption", fontsize=10, color=INK2)
    ax.tick_params(labelsize=9, colors=INK2)
    ax.grid(axis="y", color="#ecebe7", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    return ax


def log_axes(ax, ymin=1e-3):
    """The absorption-axis style of the multi-row figures (A by CH4, B by pressure)."""
    ticks = [t for t in [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100] if t >= ymin]
    ax.set_yscale("log")
    ax.set_ylim(ymin, 600)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{fr(t, 'g')} %" for t in ticks])
    ax.minorticks_off()
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fr(v, ".1f")))
    ax.tick_params(labelsize=9, colors=INK2)
    ax.grid(axis="y", color="#ecebe7", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")


def finish(fig, ax, name, band_in_legend=False):
    handles, labels = ax.get_legend_handles_labels()
    if band_in_legend:  # when the curves leave no room to label the band inside the plot
        handles.append(Patch(color=BAND, lw=0))
        labels.append("plage visée 1–20 %")
    fig.legend(handles, labels, fontsize=9.5, loc="center left", bbox_to_anchor=(0.40, 0.575),
               ncol=len(handles), frameon=False, handlelength=2.2, columnspacing=1.3)
    fig.savefig(OUT / name, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {OUT / name}")


def criteria(cfg, interferers):
    """The criteria rows, the same in both studies (the SubOcean validation case's)."""
    return [
        (None, "Critères", None),
        ("Absorption au pic", f"1–{pct(1 - cfg.transmission_min)}  (SNR {cfg.snr_min:.0f} / {cfg.spectrum_snr:.0f})",
         "SubOcean"),
        ("Séparation", f"≥ {cfg.n_separation:.0f} largeurs de raie", "SubOcean"),
        ("Interférence", f"≤ {pct(cfg.interference_max)}", "SubOcean"),
        ("Interférents", interferers, None),
    ]


# ------------------------------------------------------------- study A ------

def study_a_figure():
    cfg = study_a.config
    ppms = study_a.CH4_PPM
    # what blocks the high-CH4 cases: the same search with the interference filter off, every candidate kept
    blocked = {}
    for ppm in ppms[1:]:
        with contextlib.redirect_stdout(io.StringIO()):
            r = find_combined_windows(study_a.targets_for(ppm), dataclasses.replace(
                cfg, rank_by_interference="none", max_windows=cfg.max_stage1_candidates), study_a.run.background)
        worst = [max(c.interference.values()) for c in r.windows]
        ch4 = [1 - c.transmission["CH4"] for c in r.windows if c.transmission_ok["CH4"]]
        blocked[ppm] = (min(worst), min(ch4), max(ch4))
    ref = next(row for row in csv.DictReader(open(ROOT / "results" / "subocean_ch4_10ppm" / "windows.csv")))
    ref_interf = max(float(row["interference_ratio"]) for row in
                     csv.DictReader(open(ROOT / "results" / "subocean_ch4_10ppm" / "windows.csv"))
                     if row["window"] == ref["window"])

    rows = [
        ("Gaz mesurés", "CH4, N2O 1 ppm, H2O 30 %", None),
        ("Concentration de CH4", " · ".join(num(p) for p in ppms) + " ppm", "testée"),
        ("Ring-down", f"{cfg.L_cm / 3e10 * 1e6:.0f} µs → L = c·τ = {fr(cfg.L_cm / 1e5)} km", "SubOcean"),
        ("Pression", f"{cfg.P * 1013.25:.0f} mbar", "SubOcean"),
        ("Température", f"{cfg.T - 273.15:.0f} °C", "SubOcean"),
        ("Balayage laser", f"{fr(cfg.delta_sigma, '.2f')} cm⁻¹", "SubOcean"),
        ("Recherche (accord du laser DFB)", f"{fr(cfg.sigma_min, '.2f')}–{fr(cfg.sigma_max, '.2f')} cm⁻¹  "
                                            f"({cfg.sigma_max - cfg.sigma_min:.0f} cm⁻¹)", "SubOcean"),
    ] + criteria(cfg, "air standard (42 gaz) + isotopes")
    tiles = []
    for ppm in ppms:
        n = n_windows(f"subocean_ch4_{ppm}ppm")
        note = ("la fenêtre SubOcean\n(1re et seule)" if n else
                f"au mieux {pct(blocked[ppm][0])}\nd'interférence")
        tiles.append((f"CH4 {num(ppm)} ppm", n, note))
    fig = page("Étude A · SubOcean avec beaucoup de CH4",
               "Avec le même laser DFB, peut-on encore mesurer CH4, N2O et H2O quand CH4 monte à 1000 ou 10 000 ppm ?",
               rows, "SubOcean : valeur de l'instrument\ntestée : on balaye", "Fenêtres trouvées selon CH4", tiles)

    # zoom on the SubOcean window (rank 1 at 10 ppm), as study B zooms on its best window
    with contextlib.redirect_stdout(io.StringIO()):
        r = find_combined_windows(study_a.targets_for(ppms[0]), dataclasses.replace(cfg, max_windows=1),
                                  study_a.run.background)
    c, spectra = r.windows[0], r.window_spectra[0]
    grid = next(iter(spectra.values()))[0]
    ch4_table = ensure_lines("CH4", 2000, 9000)
    ch4_high = sum(absorption_coefficient(ch4_table, molecule_id("CH4"), i, ppms[1] * 1e-6, grid, cfg.T, cfg.P, 1e-30)
                   for i in (1, 2))  # CH4 + 13CH4 at 1000 ppm
    covered = {name: float(np.interp(c.peak_position[name], grid, ch4_high) / np.interp(c.peak_position[name], *spectra[name]))
               for name in spectra if name != "CH4"}

    notes = (f"10 ppm : les trois gaz dans la plage · interférence ≤ {fr(100 * ref_interf)} % · "
             f"pas de saturation (transmission totale ≥ {fr(c.window_min_transmission, '.2f')})\n"
             f"{num(ppms[1])} ppm (pointillé) : interférence de CH4 "
             + ", ".join(f"{pct(v)} sur {n}" for n, v in covered.items())
             + f" · ailleurs sur les 7 cm⁻¹ : ≥ {pct(blocked[ppms[1]][0])}")
    ax = spectrum_axes(fig, f"Fenêtre SubOcean : {fr(c.center)} cm⁻¹  (λ ≈ {fr(1e4 / c.center, '.3f')} µm)",
                       notes, c.lo, c.hi, 1e-3, "nombre d'onde (cm⁻¹) — balayage du laser")
    ax.text(c.lo + 0.008, 4.2, "plage visée 1–20 %", fontsize=9, color=INK2, va="center")
    ax.plot(grid, absorb(ch4_high, cfg.L_cm), color=SLOT[0], lw=1.4, ls="--", alpha=0.6,
            label=f"CH4 + ¹³CH4 à {num(ppms[1])} ppm")
    for (name, (g, a)), color in zip(spectra.items(), SLOT):
        label = {"CH4": f"CH4 {ppms[0]} ppm", "N2O": "N2O 1 ppm", "H2O": "H2O 30 %"}.get(name, name)
        ax.plot(g, absorb(a, cfg.L_cm), color=color, lw=2, label=label)
        pos, peak = c.peak_position[name], 100 * (1 - c.transmission[name])
        ax.plot(pos, peak, "o", ms=9, mfc="none", mec=color, mew=2)
        edge = pos > c.hi - 0.04  # CH4's peak sits at the right edge of the scan
        ax.annotate(f"{name}\n{fr(peak)} %", (pos, peak), xytext=(-8 if edge else 0, 9), textcoords="offset points",
                    ha="right" if edge else "center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    finish(fig, ax, "study_a.png")


def study_a_ch4_figure():
    """Why CH4 decides study A: the SubOcean window at every tested CH4 level, over the whole
    DFB tuning range (left) and what the laser scan sees (right) -- the layout of study_b_pressure."""
    cfg = study_a.config
    ppms = study_a.CH4_PPM
    with contextlib.redirect_stdout(io.StringIO()):
        r = find_combined_windows(study_a.targets_for(ppms[0]), dataclasses.replace(cfg, max_windows=1),
                                  study_a.run.background)
    c = r.windows[0]  # the SubOcean window, rank 1 at 10 ppm; its peaks stay fixed across rows
    wide = np.arange(cfg.sigma_min, cfg.sigma_max, cfg.wavenumber_step)
    label = {"CH4": "CH4 (" + " · ".join(num(p) for p in ppms) + " ppm)", "N2O": "N2O 1 ppm", "H2O": "H2O 30 %"}

    fig, axes = plt.subplots(len(ppms), 2, figsize=(13, 8.6), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.25, 1], hspace=0.3, wspace=0.1,
                                              left=0.22, right=0.985, top=0.8, bottom=0.08))
    fig.text(0.02, 0.97, "Étude A · la même fenêtre à trois concentrations de CH4", fontsize=17,
             fontweight="bold", color=INK, va="center")
    fig.text(0.02, 0.91, f"Fenêtre SubOcean {fr(c.center)} cm⁻¹ (la 1re à {ppms[0]} ppm), N2O 1 ppm, H2O 30 %, "
             f"{cfg.P * 1013.25:.0f} mbar, L = {fr(cfg.L_cm / 1e5)} km.\nPlus il y a de CH4, plus ses raies "
             "(et celles de ¹³CH4) montent sous les pics de N2O et H2O.", fontsize=11, color=INK2, va="center",
             linespacing=1.5)
    for ppm, (ax_wide, ax_zoom) in zip(ppms, axes):
        targets = study_a.targets_for(ppm)
        own = {t.label: absorption_coefficient(ensure_lines(t.name, 2000, 9000), molecule_id(t.name), t.iso_id,
                                               t.mole_fraction, wide, cfg.T, cfg.P, 1e-30) for t in targets}
        window = SimpleNamespace(peak_position=c.peak_position,
                                 peak_alpha={n: float(np.interp(c.peak_position[n], wide, a)) for n, a in own.items()})
        with contextlib.redirect_stdout(io.StringIO()):
            _, others, ratio, main = compute_interference(
                window, wide, interferer_list(targets, study_a.run.background), cfg, own)
        peak = {n: absorb(a, cfg.L_cm) for n, a in window.peak_alpha.items()}
        for ax in (ax_wide, ax_zoom):
            ax.axhspan(1, 20, color=BAND, lw=0, zorder=0)
            ax.plot(wide, absorb(sum(others.values()), cfg.L_cm), color=MUTED, lw=1.1, ls="--",
                    label="autres (¹³CH4, HDO, air…)")
            for (name, a), color in zip(own.items(), SLOT):
                ax.plot(wide, absorb(a, cfg.L_cm), color=color, lw=1.6 if ax is ax_wide else 1.8, label=label[name])
            log_axes(ax)
        ax_zoom.set_yticklabels([])
        ax_wide.set_xlim(wide[0], wide[-1])
        scan = ax_wide.axvspan(c.lo, c.hi, color=MUTED, alpha=0.18, lw=0, zorder=0)
        ax_zoom.set_xlim(c.lo, c.hi)
        for name, color in zip(own, SLOT):
            ax_zoom.plot(c.peak_position[name], peak[name], "o", ms=8, mfc="none", mec=color, mew=1.8)

        top_pct = 100 * (1 - cfg.transmission_min)
        too_much = [n for n, v in ratio.items() if v > cfg.interference_max]
        out_of_range = [f"{n} à {fr(v, '.0f')} %" for n, v in peak.items() if not 1 <= v <= top_pct]
        problems = ([f"interférence > {pct(cfg.interference_max)} sur {' et '.join(too_much)}"] if too_much else []) \
            + ([f"{', '.join(out_of_range)} au pic : hors 1–{fr(top_pct, '.0f')} %"] if out_of_range else [])
        verdict = ("éliminée : " + "\n".join(problems)) if problems else \
            "passe : interférence ok, les trois gaz dans la plage"
        ax_zoom.text(0.01, 0.98, verdict, transform=ax_zoom.transAxes, fontsize=10, fontweight="bold",
                     color=INK, va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d6d5d0", lw=0.8))
        mid = (ax_wide.get_position().y0 + ax_wide.get_position().y1) / 2
        fig.text(0.02, mid + 0.055, f"CH4 {num(ppm)} ppm", fontsize=14, fontweight="bold", color=INK, va="center")
        fig.text(0.02, mid - 0.015, "interférence au pic :\n" + "\n".join(
                 f"  {n} : {fr(100 * v) if v < 0.1 else fr(100 * v, '.0f')} %  ({NICE.get(main[n], main[n])})"
                 for n, v in ratio.items()), fontsize=9.5, color=INK2, va="center", linespacing=1.5)
    axes[0][0].set_title("toute la plage d'accord du laser DFB (7 cm⁻¹)", fontsize=10.5, color=INK2, loc="left")
    axes[0][1].set_title("ce que voit le laser (zoom sur le balayage)", fontsize=10.5, color=INK2, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("nombre d'onde (cm⁻¹)", fontsize=10, color=INK2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles += [Patch(color=BAND, lw=0), scan]
    labels += ["plage visée 1–20 %", "balayage du laser (zoom à droite)"]
    fig.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], fontsize=9.5, loc="center left",
               bbox_to_anchor=(0.015, 0.845), ncol=6, frameon=False, handlelength=2.0, columnspacing=1.1)
    fig.savefig(OUT / "study_a_ch4.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {OUT / 'study_a_ch4.png'}")


# ------------------------------------------------------------- study B ------

def study_b_figure():
    cfg = study_b.run.config
    rows = [
        ("Échantillon", "100 % vapeur d'eau", "SWIS"),
        ("Ring-down", f"{fr(study_b.RINGDOWN_S * 1e6, '.0f')} µs → L = c·τ = {fr(study_b.L_CM / 1e5)} km", "SWIS"),
        ("Gaz mesurés", "H₂¹⁶O, H₂¹⁸O, HDO  (δ¹⁸O, δD)", None),
        ("Pression", " · ".join(f"{p}" for p in study_b.P_MBAR) + " mbar", "testée"),
        ("Température", f"{cfg.T - 273.15:.0f} °C", "SubOcean"),
        ("Balayage laser", f"{fr(cfg.delta_sigma, '.2f')} cm⁻¹", "SubOcean"),
        ("Recherche", f"{cfg.sigma_min:.0f}–{cfg.sigma_max:.0f} cm⁻¹", None),
    ] + criteria(cfg, "autres isotopes de l'eau, pas d'air")
    tiles = []
    for p in study_b.P_MBAR:
        n = n_windows(f"swis_{p}mbar")
        note = ("HDO et H₂¹⁸O sous les\nailes de H₂¹⁶O" if n == 0 else
                f"fenêtre{'s' if n > 1 else ''} sous {pct(cfg.interference_max)}\nd'interférence")
        tiles.append((f"{p} mbar", n, note))
    fig = page("Étude B · isotopes de l'eau pour SWIS",
               "Peut-on mesurer H₂¹⁶O, H₂¹⁸O et HDO dans un même balayage laser, dans 100 % de vapeur d'eau ?",
               rows, "SWIS : valeur de l'instrument · SubOcean : reprise,\nà confirmer · testée : inconnue, on balaye",
               "Fenêtres trouvées selon la pression", tiles)

    p = min(study_b.P_MBAR)
    with contextlib.redirect_stdout(io.StringIO()):
        r = find_combined_windows(study_b.targets, dataclasses.replace(cfg, L_cm=study_b.L_CM, P=p / 1013.25,
                                                                       max_windows=1), study_b.background)
    c, spectra, others = r.windows[0], r.window_spectra[0], r.interferer_spectra[0]
    ratio = c.ratio_temperature_error
    notes = (f"Les trois isotopes dans la plage · interférence ≤ {fr(100 * max(c.interference.values()))} % · "
             f"pas de saturation (transmission totale ≥ {fr(c.window_min_transmission, '.2f')})\n"
             f"À surveiller : rapports sensibles à T — δ¹⁸O {fr(1e3 * ratio['H2O/H2(18O)'])} ‰ et "
             f"δD {fr(1e3 * ratio['H2O/HD(16O)'])} ‰ pour 0,1 K (seuil 1 ‰)")
    ax = spectrum_axes(fig, f"Meilleure fenêtre à {p} mbar : {fr(c.center)} cm⁻¹  (λ ≈ {fr(1e4 / c.center, '.3f')} µm)",
                       notes, c.lo, c.hi, 1e-4, "nombre d'onde (cm⁻¹) — balayage du laser")
    ax.text(c.lo + 0.008, 4.2, "plage visée 1–20 %", fontsize=9, color=INK2, va="center")
    top = next((i for i in c.interferers if i.label in others), None)
    if top is not None:
        grid, alpha = others[top.label]
        ax.plot(grid, absorb(alpha, study_b.L_CM), color=MUTED, lw=1.3, ls="--",
                label=f"{NICE.get(top.label, top.label)} (interférent)")
    for (name, (grid, alpha)), color in zip(spectra.items(), SLOT):
        ax.plot(grid, absorb(alpha, study_b.L_CM), color=color, lw=2, label=NICE[name])
        pos, a = c.peak_position[name], 100 * (1 - c.transmission[name])
        ax.plot(pos, a, "o", ms=9, mfc="none", mec=color, mew=2)
        ax.annotate(f"{NICE[name]}\n{fr(a)} %", (pos, a), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    finish(fig, ax, "study_b.png")


def study_b_pressure_figure():
    """Why pressure decides study B: the best 5 mbar window at every tested pressure, with the
    strong H2(16)O lines around it (left) and what the laser scan sees (right)."""
    cfg = study_b.run.config
    p_best = min(study_b.P_MBAR)
    with contextlib.redirect_stdout(io.StringIO()):
        r = find_combined_windows(study_b.targets, dataclasses.replace(cfg, L_cm=study_b.L_CM, P=p_best / 1013.25,
                                                                       max_windows=1), study_b.background)
    c = r.windows[0]
    wide = np.arange(c.center - 3.3, c.center + 2.0, cfg.wavenumber_step)
    table = ensure_lines("H2O", 2000, 9000)
    isos = [t.iso_id for t in study_b.targets]
    names = list(c.peak_position)  # same order as isos
    minor = [i for i in range(1, 8) if i not in isos]

    fig, axes = plt.subplots(len(study_b.P_MBAR), 2, figsize=(13, 8.6), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.25, 1], hspace=0.3, wspace=0.1,
                                              left=0.19, right=0.985, top=0.8, bottom=0.08))
    fig.text(0.02, 0.97, "Étude B · la même fenêtre à trois pressions", fontsize=17, fontweight="bold",
             color=INK, va="center")
    fig.text(0.02, 0.91, f"Fenêtre {fr(c.center)} cm⁻¹ (la meilleure à {p_best} mbar), 100 % de vapeur d'eau, "
             f"L = {fr(study_b.L_CM / 1e5)} km.\nPlus la pression baisse, plus les raies sont fines : les ailes de "
             "H₂¹⁶O ne passent plus sous HDO et H₂¹⁸O.", fontsize=11, color=INK2, va="center", linespacing=1.5)
    for P, (ax_wide, ax_zoom) in zip(study_b.P_MBAR, axes):
        a = {i: absorption_coefficient(table, molecule_id("H2O"), i, 1.0, wide, cfg.T, P / 1013.25, 1e-30)
             for i in range(1, 8)}
        total = sum(a.values())
        interference = {}
        for name, i in zip(names, isos):
            near = np.where(np.abs(wide - c.peak_position[name]) < 0.015)[0]
            k = near[np.argmax(a[i][near])]
            interference[name] = ((total[k] - a[i][k]) / a[i][k], absorb(a[i][k], study_b.L_CM), wide[k])
        for ax in (ax_wide, ax_zoom):
            ax.axhspan(1, 20, color=BAND, lw=0, zorder=0)
            ax.plot(wide, absorb(sum(a[i] for i in minor), study_b.L_CM), color=MUTED, lw=1.1, ls="--",
                    label="autres isotopes (H₂¹⁷O, HD¹⁸O…)")
            for name, i, color in zip(names, isos, SLOT):
                ax.plot(wide, absorb(a[i], study_b.L_CM), color=color, lw=1.8, label=NICE[name])
            log_axes(ax)
        ax_zoom.set_yticklabels([])
        ax_wide.set_xlim(wide[0], wide[-1])
        scan = ax_wide.axvspan(c.lo, c.hi, color=MUTED, alpha=0.18, lw=0, zorder=0)
        ax_zoom.set_xlim(c.lo, c.hi)
        for name, color in zip(names, SLOT):
            ratio, peak, pos = interference[name]
            ax_zoom.plot(pos, peak, "o", ms=8, mfc="none", mec=color, mew=1.8)
        too_much = [NICE[n] for n, v in interference.items() if v[0] > cfg.interference_max]
        top_pct = 100 * (1 - cfg.transmission_min)
        out_of_range = [f"{NICE[n]} à {fr(v[1], '.0f')} %" for n, v in interference.items()
                        if not 1 <= v[1] <= top_pct]
        mid = (ax_wide.get_position().y0 + ax_wide.get_position().y1) / 2
        fig.text(0.02, mid + 0.055, f"{P} mbar", fontsize=15, fontweight="bold", color=INK, va="center")
        fig.text(0.02, mid - 0.015, "interférence au pic :\n" + "\n".join(
                 f"  {NICE[n]} : {fr(100 * v[0])} %" for n, v in interference.items()),
                 fontsize=9.5, color=INK2, va="center", linespacing=1.5)
        if too_much:
            verdict = f"éliminée : interférence > {pct(cfg.interference_max)} sur {' et '.join(too_much)}"
        elif out_of_range:
            verdict = (f"interférence ok, mais {', '.join(out_of_range)} : au-dessus de "
                       f"{fr(top_pct, '.0f')} %")
        else:
            verdict = "passe : interférence ok, les trois gaz dans la plage"
        ax_zoom.text(0.01, 0.98, verdict, transform=ax_zoom.transAxes, fontsize=10, fontweight="bold",
                     color=INK, va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d6d5d0", lw=0.8))
    axes[0][0].set_title("autour de la fenêtre : les raies fortes de H₂¹⁶O (saturées)", fontsize=10.5,
                         color=INK2, loc="left")
    axes[0][1].set_title("ce que voit le laser (zoom sur le balayage)", fontsize=10.5, color=INK2, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("nombre d'onde (cm⁻¹)", fontsize=10, color=INK2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles += [Patch(color=BAND, lw=0), scan]
    labels += ["plage visée 1–20 %", "balayage du laser (zoom à droite)"]
    fig.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], fontsize=9.5, loc="center left",
               bbox_to_anchor=(0.015, 0.845), ncol=6, frameon=False, handlelength=2.0, columnspacing=1.1)
    fig.savefig(OUT / "study_b_pressure.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {OUT / 'study_b_pressure.png'}")


if __name__ == "__main__":
    study_a_figure()
    study_a_ch4_figure()
    study_b_figure()
    study_b_pressure_figure()
