"""Text report, CSV export and quick-look plots for a CombinedSearchResult."""
from __future__ import annotations

import csv
import os

from .windows import CombinedSearchResult

_TOP_INTERFERERS_TEXT = 5  # listed per window in the console report
_TOP_INTERFERERS_PLOT = 5  # drawn per window in the plots...
_PLOT_INTERFERER_MIN = 1e-3  # ...if their peak is >= this fraction of the weakest target's alpha
# Interference ratio shown in red (plot) / with '!' (console) above this. Display only.
_INTERFERENCE_WARN = 0.01
_RED = "#c22b2b"


def _pct(x: float) -> str:
    return f"{100 * x:,.0f}%" if x >= 1 else f"{100 * x:.2g}%"


def print_combined_report(result: CombinedSearchResult, names: list | None = None) -> None:
    """result: as returned by find_combined_windows()."""
    windows = result.windows
    if not windows:
        if result.n_rejected_saturated or result.n_rejected_interference:
            print(f"No window left: candidates had every target distinguishable, but "
                  f"{result.n_rejected_interference} were dropped for interference (> interference_max) "
                  f"and {result.n_rejected_saturated} as saturated (window_transmission_min).")
        else:
            print("No window found where all targets have a mutually distinguishable line "
                  "(separation requirement never satisfied).")
        return

    names = names or list(windows[0].peak_alpha)
    tmin, tmax = result.transmission_range
    print(f"Top {len(windows)} combined window(s) for {' + '.join(names)}")
    if tmin is not None:
        print(f"acceptable I/I0 range: [{tmin:.3f}, {tmax:.3f}]  "
              f"(absorption {100 * (1 - tmax):.2g}-{100 * (1 - tmin):.2g} %)\n")
    header = (
        f"{'#':>2} {'center':>10} {'window':>17} "
        + " ".join(f"{n + ' peak':>14}" for n in names)
        + " " + " ".join(f"{n + ' I/I0':>10}" for n in names)
        + f" {'in range':>9}  worst pair"
    )
    print(header)
    print("-" * len(header))
    for i, c in enumerate(windows, 1):
        peaks = " ".join(f"{c.peak_alpha[n]:>14.3e}" for n in names)
        # Out-of-range values are wrapped in [brackets] -- reported, not
        # filtered, so a run can still return its best candidates even when
        # none reach [transmission_min, transmission_max]. Brackets read much
        # more clearly in a scan than a trailing '*' does.
        trans = " ".join(
            (f"[{c.transmission[n]:.3f}]" if not c.transmission_ok[n] else f" {c.transmission[n]:.3f} ").rjust(10)
            for n in names
        )
        n_ok = sum(c.transmission_ok.values())
        in_range_str = f"{n_ok}/{len(names)}"
        pair_str = f"{c.worst_pair[0]}/{c.worst_pair[1]}" if c.worst_pair else "-"  # full labels:
        # truncating turned isotopologues like '(13C)H4' into an unreadable '(1' 
        print(
            f"{i:>2} {c.center:>10.3f} [{c.lo:>7.3f},{c.hi:>7.3f}] {peaks} {trans} "
            f"{in_range_str:>9}  {pair_str}"
        )
    print(
        "\nunits: center/window in cm^-1, peaks = alpha at each species' selected peak in cm^-1,"
        "\n       I/I0 = exp(-alpha*L_cm) = transmission (0-1),"
        "\n       [bracketed] I/I0 = outside [transmission_min, transmission_max] (informational, not filtered)."
        "\n       in range = how many of the species are inside that range, e.g. '1/3'."
        "\n       worst pair = tightest species pair in the window (every listed window already passed the"
        "\n       separation check -- that part IS a hard filter, so there's no pass/fail column for it)."
    )

    if result.n_rejected_interference:
        print(f"\n({result.n_rejected_interference} better-ranked window(s) dropped: a target's "
              f"interference above interference_max.)")
    if result.n_rejected_saturated:
        print(f"({result.n_rejected_saturated} better-ranked window(s) dropped: total I/I0 somewhere in "
              f"the window below window_transmission_min -- saturated.)")

    print("\n=== Per-window detail: selected peaks, temperature sensitivity, interferers ===")
    for i, c in enumerate(windows, 1):
        _print_window_detail(i, c, names)
    print(
        "\npeak      = top of the species' real curve in the window (between two of its lines if\n"
        "            they overlap) -- the point alpha, I/I0, T coeff and interference are read at.\n"
        "T coeff   = (alpha(T+dT) - alpha(T)) / alpha(T) / dT at that peak, constant P (%/K).\n"
        "interf.   = sum of every other species' alpha at that peak / the target's own alpha there\n"
        "            ('!' above 1%); main = the biggest single contributor. Other species = the\n"
        "            other targets, the targets' own minor isotopologues, and atm_standard.csv\n"
        "            gases at their standard concentrations. Not used for ranking."
    )


def _print_window_detail(i: int, c, names: list) -> None:
    print(f"\n#{i}  center {c.center:.3f} cm^-1  [{c.lo:.3f}, {c.hi:.3f}]   "
          f"lowest total I/I0 {c.window_min_transmission:.3f} at {c.window_min_position:.4f}")
    print(f"    {'species':<14}{'peak':>12}{'alpha':>11}{'I/I0':>8}{'T coeff':>10}{'interf.':>10}  main interferer")
    for n in names:
        tc = c.temperature_coeff.get(n)
        tc_str = f"{tc:+.3f}" if tc is not None else "-"
        itf = c.interference.get(n)
        itf_str = (_pct(itf) + ("!" if itf > _INTERFERENCE_WARN else " ")) if itf is not None else "-"
        print(f"    {n:<14}{c.peak_position[n]:>12.4f}{c.peak_alpha[n]:>11.2e}{c.transmission[n]:>8.3f}"
              f"{tc_str:>10}{itf_str:>10}  {c.main_interferer.get(n) or '-'}")
        if not c.peak_is_window_max[n]:
            ratio = c.window_max_alpha[n] / c.peak_alpha[n]
            print(f"    {'':<14}! its alpha is {ratio:.1f}x higher at {c.window_max_position[n]:.4f} "
                  f"({_why_not_max(c, n)})")
    for pair, coeff in c.ratio_temperature_coeff.items():
        print(f"    ratio {pair}: T coeff {coeff:+.3f} %/K  {_ratio_error_str(c, pair)}")
    if c.interferers:
        top = " | ".join(
            f"{r.label} {r.peak_alpha:.1e} @{r.peak_position:.3f} (I/I0 {r.transmission:.4f})"
            for r in c.interferers[:_TOP_INTERFERERS_TEXT]
        )
        print(f"    top interferers: {top}")


_NOT_MAX_REASON = {  # CombinedWindowResult.peak_note -> text
    "edge": "wing of a line outside the window",
    "range": "taller peak inside, not used: it absorbs outside the target range",
    "separation": "taller peak inside, not used: too close to another target",
    "width": "taller peak inside, not used: its width can't be measured in this window",
}


def _ratio_error_str(c, pair: str) -> str:
    """'-> 0.32 permil for the cell's T stability' (+ '!' if above the target precision)."""
    err = c.ratio_temperature_error.get(pair)
    if err is None:
        return ""
    return f"-> {1e3 * err:.2g} permil at the cell's T stability" + ("" if c.ratio_temperature_ok[pair] else "  !")


def _why_not_max(c, n: str) -> str:
    return _NOT_MAX_REASON.get(c.peak_note.get(n), "stronger elsewhere in the window")


def save_results_csv(result: CombinedSearchResult, output_dir: str) -> list:
    """Write windows.csv (one row per window x target), interferers.csv (window x
    interferer) and temperature.csv (window x target x T) into output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    paths = [os.path.join(output_dir, f) for f in ("windows.csv", "interferers.csv", "temperature.csv")]

    with open(paths[0], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window", "center", "lo", "hi", "species", "peak_position", "peak_alpha", "fwhm",
                    "tau", "transmission", "transmission_ok", "window_max_alpha", "window_max_position",
                    "peak_is_window_max", "temperature_coeff_pct_per_K", "interference_ratio",
                    "main_interferer", "window_min_transmission", "window_min_position"])
        for i, c in enumerate(result.windows, 1):
            for n in c.peak_alpha:
                w.writerow([i, c.center, c.lo, c.hi, n, c.peak_position[n], c.peak_alpha[n], c.fwhm[n],
                            c.tau[n], c.transmission[n], c.transmission_ok[n], c.window_max_alpha[n],
                            c.window_max_position[n], c.peak_is_window_max[n],
                            c.temperature_coeff.get(n), c.interference.get(n), c.main_interferer.get(n),
                            c.window_min_transmission, c.window_min_position])

    with open(paths[1], "w", newline="") as f:
        w = csv.writer(f)
        target_names = list(result.windows[0].peak_alpha) if result.windows else []
        w.writerow(["window", "interferer", "molecule", "iso_id", "mole_fraction", "peak_alpha",
                    "peak_position", "transmission"]
                   + [f"alpha_at_{n}" for n in target_names]
                   + [f"ratio_to_{n}" for n in target_names])
        for i, c in enumerate(result.windows, 1):
            for r in c.interferers:
                w.writerow([i, r.label, r.molecule, r.iso_id, r.mole_fraction, r.peak_alpha,
                            r.peak_position, r.transmission]
                           + [r.alpha_at_target[n] for n in target_names]
                           + [r.alpha_at_target[n] / c.peak_alpha[n] for n in target_names])

    with open(paths[2], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window", "species", "T", "alpha", "relative_change_vs_T0"])
        for i, c in enumerate(result.windows, 1):
            for n, sweep in c.temperature_sweep.items():
                a0 = sweep[0][1]
                for T, a in sweep:
                    w.writerow([i, n, T, a, (a - a0) / a0 if a0 > 0 else ""])
    return paths


def plot_combined_window_detail(result: CombinedSearchResult, index: int):
    """Zoomed-in report for one CombinedWindowResult: every target species
    plotted separately, using that window's OWN spectra (each window has its
    own narrow stage-2 range -- see CombinedSearchResult docstring), each
    target's SELECTED line marked (the one the window was picked for), the
    strongest interferers overlaid, and per-species metrics annotated. No
    extra viewing margin beyond [lo, hi]: stage 2 only ever computes that
    exact narrow range, so there's no wider context available to show.

    index: 0-based position into result.windows (already sorted best-first).
    """
    import matplotlib.pyplot as plt

    c = result.windows[index]
    spectra = result.window_spectra[index]
    names = list(c.peak_alpha)

    # Annotation goes in a reserved strip BELOW the axes (figure coordinates,
    # not axes coordinates) so it never sits on top of -- and never hides --
    # any plotted curve, unlike an in-plot text box.
    n_lines = 2 + len(names) + len(c.ratio_temperature_coeff) + 2
    fig_h = 5 + 0.22 * n_lines
    fig, ax = plt.subplots(figsize=(11, fig_h))

    colors = {}
    for name, (nu_grid, alpha) in spectra.items():
        (line,) = ax.plot(nu_grid, alpha, label=name, lw=1.5)
        colors[name] = line.get_color()

    # Strongest interferers, only those big enough to matter next to the targets --
    # otherwise dozens of 1e-15 curves would crush the log axis.
    floor = _PLOT_INTERFERER_MIN * min(c.peak_alpha.values())
    int_spectra = result.interferer_spectra[index] if index < len(result.interferer_spectra) else {}
    shown = [r for r in c.interferers if r.peak_alpha >= floor][:_TOP_INTERFERERS_PLOT]
    for r, ls in zip(shown, ["--", "-.", ":", (0, (5, 2, 1, 2)), (0, (1, 1))]):
        nu_grid, alpha = int_spectra[r.label]
        ax.plot(nu_grid, alpha, color="0.45", ls=ls, lw=1.0, label=f"{r.label} (interferer)")

    ax.axvspan(c.lo, c.hi, color="tab:green", alpha=0.08, label="selected window")

    # The selected peak of each target: marker + label, in the species' own color.
    # If something else in the window is stronger for that species (a wing from a
    # line outside the window, or another line), mark that too, with an x.
    for name in names:
        pos, a = c.peak_position[name], c.peak_alpha[name]
        ax.axvline(pos, color=colors[name], ls=":", lw=0.8)
        ax.plot(pos, a, "o", ms=8, mfc="none", mec=colors[name], mew=2)
        ax.annotate(f"{name}\n{pos:.4f}", (pos, a), xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=colors[name], fontweight="bold")
        if not c.peak_is_window_max[name]:
            ax.plot(c.window_max_position[name], c.window_max_alpha[name], "x", ms=8,
                    color=colors[name], mew=2)

    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorption coefficient (cm$^{-1}$)")
    ax.set_title(f"Combined window #{index + 1} @ {c.center:.3f} cm$^{{-1}}$ -- {' + '.join(names)}")
    # Legend outside the axes, on the right: inside, it hid the selected-line labels.
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    ax.margins(y=0.15)
    # Without this, matplotlib factors the common ~4400 cm^-1 offset out of the tick labels
    # and prints it separately (e.g. ticks "-0.3 ... 0.2" plus a "+4.4e3" corner label) --
    # technically correct but unreadable; show the real wavenumber on every tick instead.
    ax.ticklabel_format(useOffset=False, axis="x")

    # axis_label_space reserves room for the x-tick labels + "Wavenumber..."
    # axis label themselves, BEFORE the annotation text starts -- without it,
    # the first annotation line collides with the tick labels.
    axis_label_space = 0.12
    text_block_h = 0.028 * n_lines
    bottom_margin = axis_label_space + text_block_h
    fig.subplots_adjust(bottom=bottom_margin, left=0.08, right=0.78)

    pair_str = f"{c.worst_pair[0]}/{c.worst_pair[1]}: {c.worst_pair[2]:.4f} (req {c.worst_pair[3]:.4f})" if c.worst_pair else "-"

    # Figure-y layout, bottom (0) to top (1): [0, text_block_h] is the
    # annotation strip; [text_block_h, bottom_margin] is left for the
    # x-tick labels and "Wavenumber..." axis label; the axes start above
    # that, at bottom_margin. Text runs top-down within its own strip only.
    line_h = (text_block_h - 0.02) / n_lines
    y = text_block_h - 0.01

    def txt(x, text, color="black", bold=False):
        fig.text(x, y, text, ha="left", va="top", fontsize=8, color=color,
                 fontweight="bold" if bold else "normal")

    note = "o = selected peak: top of the curve (alpha and I/I0 = exp(-alpha*L_cm) taken there)"
    flagged = [n for n in names if not c.peak_is_window_max[n]]
    if flagged:
        note += ";  x = same species stronger elsewhere (" + ", ".join(
            f"{n}: {_why_not_max(c, n)}" for n in flagged) + ")"
    txt(0.06, note)
    y -= line_h

    # A small table: one row per species, fixed columns. Everything is black except the
    # species name (in its curve's color, to match the plot) and a value that fails its
    # criterion, in red -- so a glance shows WHICH parameter is the problem.
    cols = [0.06, 0.17, 0.28, 0.37, 0.46, 0.57, 0.66]
    for x, h in zip(cols, ["species", "peak (cm$^{-1}$)", "alpha (cm$^{-1}$)", "I/I0",
                           "T coeff (%/K)", "interf.", "main interferer"]):
        txt(x, h, bold=True)
    y -= line_h
    for n in names:
        tc = c.temperature_coeff.get(n)
        itf = c.interference.get(n)
        txt(cols[0], n, color=colors[n], bold=True)
        txt(cols[1], f"{c.peak_position[n]:.4f}")
        txt(cols[2], f"{c.peak_alpha[n]:.2e}")
        txt(cols[3], f"{c.transmission[n]:.3f}", color="black" if c.transmission_ok[n] else _RED,
            bold=not c.transmission_ok[n])
        txt(cols[4], f"{tc:+.3f}" if tc is not None else "-")
        bad_itf = itf is not None and itf > _INTERFERENCE_WARN
        txt(cols[5], _pct(itf) if itf is not None else "-", color=_RED if bad_itf else "black", bold=bad_itf)
        txt(cols[6], c.main_interferer.get(n) or "-")
        y -= line_h
    for pair, coeff in c.ratio_temperature_coeff.items():
        ok = c.ratio_temperature_ok.get(pair, True)
        txt(0.06, f"ratio {pair}: T coeff {coeff:+.3f} %/K  {_ratio_error_str(c, pair)}",
            color="black" if ok else _RED, bold=not ok)
        y -= line_h
    txt(0.06, f"lowest total I/I0 in the window (all species): {c.window_min_transmission:.3f} "
              f"at {c.window_min_position:.4f}")
    y -= line_h
    txt(0.06, f"tightest pair: {pair_str}")

    return fig


def save_combined_window_details(result: CombinedSearchResult, output_dir: str, **kwargs) -> list:
    """Save one PNG per window (plot_combined_window_detail) into output_dir."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i in range(len(result.windows)):
        fig = plot_combined_window_detail(result, i, **kwargs)
        path = os.path.join(output_dir, f"combined_window_{i + 1:02d}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
