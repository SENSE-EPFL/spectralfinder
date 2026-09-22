"""Text report and quick-look plots for a CombinedSearchResult."""
from __future__ import annotations

from .windows import CombinedSearchResult


def print_combined_report(result: CombinedSearchResult, names: list | None = None) -> None:
    """result: as returned by find_combined_windows()."""
    windows = result.windows
    if not windows:
        print("No window found where all targets have a mutually distinguishable line "
              "(separation requirement never satisfied).")
        return

    names = names or list(windows[0].peak_alpha)
    print(f"Top {len(windows)} combined window(s) for {' + '.join(names)}\n")
    header = (
        f"{'#':>2} {'center':>10} {'window':>17} "
        + " ".join(f"{n + ' peak':>14}" for n in names)
        + " " + " ".join(f"{n + ' I/I0':>10}" for n in names)
        + f" {'in range':>9} {'worst pair':>10}"
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
        pair_str = f"{c.worst_pair[0][:2]}/{c.worst_pair[1][:2]}" if c.worst_pair else "-"
        print(
            f"{i:>2} {c.center:>10.3f} [{c.lo:>7.3f},{c.hi:>7.3f}] {peaks} {trans} "
            f"{in_range_str:>9} {pair_str:>10}"
        )
    print(
        "\nunits: center/window in cm^-1, peaks in cm^-1, I/I0 = exp(-alpha*L_cm) = transmission (0-1),"
        "\n       [bracketed] I/I0 = outside [transmission_min, transmission_max] (informational, not filtered)."
        "\n       in range = how many of the species are inside that range, e.g. '1/3'."
        "\n       worst pair = tightest species pair in the window (every listed window already passed the"
        "\n       separation check -- that part IS a hard filter, so there's no pass/fail column for it)."
    )


def plot_combined_window_detail(result: CombinedSearchResult, index: int):
    """Zoomed-in report for one CombinedWindowResult: every target species
    plotted separately, using that window's OWN spectra (each window has its
    own narrow stage-2 range -- see CombinedSearchResult docstring), with the
    tightest pair and per-species metrics annotated. No extra viewing margin
    beyond [lo, hi]: stage 2 only ever computes that exact narrow range, so
    there's no wider context available to show.

    index: 0-based position into result.windows (already sorted best-first).
    """
    import matplotlib.pyplot as plt

    c = result.windows[index]
    spectra = result.window_spectra[index]
    names = list(c.peak_alpha)

    # Annotation goes in a reserved strip BELOW the axes (figure coordinates,
    # not axes coordinates) so it never sits on top of -- and never hides --
    # any plotted curve, unlike an in-plot text box.
    n_lines = 2 + len(names) + 1
    fig_h = 5 + 0.22 * n_lines
    fig, ax = plt.subplots(figsize=(8, fig_h))

    for name, (nu_grid, alpha) in spectra.items():
        ax.plot(nu_grid, alpha, label=name, lw=1.5)

    ax.axvspan(c.lo, c.hi, color="tab:green", alpha=0.12, label="selected window")
    for name, pos in c.line_position.items():
        ax.axvline(pos, color="0.6", ls=":", lw=0.8)

    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorption coefficient (cm$^{-1}$)")
    ax.set_title(f"Combined window #{index + 1} @ {c.center:.3f} cm$^{{-1}}$ -- {' + '.join(names)}")
    ax.legend(loc="upper right", fontsize=8)

    # axis_label_space reserves room for the x-tick labels + "Wavenumber..."
    # axis label themselves, BEFORE the annotation text starts -- without it,
    # the first annotation line collides with the tick labels.
    axis_label_space = 0.12
    text_block_h = 0.028 * n_lines
    bottom_margin = axis_label_space + text_block_h
    fig.subplots_adjust(bottom=bottom_margin)

    peaks_str = ", ".join(f"{n}={c.peak_alpha[n]:.2e}" for n in names)
    pair_str = f"{c.worst_pair[0]}/{c.worst_pair[1]}: {c.worst_pair[2]:.4f} (req {c.worst_pair[3]:.4f})" if c.worst_pair else "-"

    # Figure-y layout, bottom (0) to top (1): [0, text_block_h] is the
    # annotation strip; [text_block_h, bottom_margin] is left for the
    # x-tick labels and "Wavenumber..." axis label; the axes start above
    # that, at bottom_margin. Text runs top-down within its own strip only.
    line_h = (text_block_h - 0.02) / n_lines
    y = text_block_h - 0.01
    fig.text(0.06, y, f"peaks (cm$^{{-1}}$): {peaks_str}", ha="left", va="top", fontsize=8)
    y -= line_h
    fig.text(0.06, y, "I/I0 = exp(-alpha*L_cm):", ha="left", va="top", fontsize=8)
    y -= line_h
    for n in names:
        ok = c.transmission_ok[n]
        color = "#1a7d3a" if ok else "#c22b2b"
        status = "ok" if ok else "OUT OF RANGE"
        fig.text(0.09, y, f"{n} = {c.transmission[n]:.3f}  {status}",
                 ha="left", va="top", fontsize=8, color=color, fontweight="bold")
        y -= line_h
    fig.text(0.06, y, f"tightest pair: {pair_str}", ha="left", va="top", fontsize=8)

    return fig


def save_combined_window_details(result: CombinedSearchResult, output_dir: str, **kwargs) -> list:
    """Save one PNG per window (plot_combined_window_detail) into output_dir."""
    import os

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
