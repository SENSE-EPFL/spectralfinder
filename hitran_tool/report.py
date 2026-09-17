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
        + " " + " ".join(f"{n + ' tau':>10}" for n in names)
        + f" {'worst pair':>10}"
    )
    print(header)
    print("-" * len(header))
    for i, c in enumerate(windows, 1):
        peaks = " ".join(f"{c.peak_alpha[n]:>14.3e}" for n in names)
        # '*' flags a species whose tau falls outside [tau_min, tau_max] at
        # this L_cm -- reported, not filtered, so a run can still return its
        # best candidates even when none of them reach the acceptable range.
        taus = " ".join(f"{c.tau[n]:>9.3f}{'' if c.tau_ok[n] else '*'}" for n in names)
        pair_str = f"{c.worst_pair[0][:2]}/{c.worst_pair[1][:2]}" if c.worst_pair else "-"
        print(
            f"{i:>2} {c.center:>10.3f} [{c.lo:>7.3f},{c.hi:>7.3f}] {peaks} {taus} "
            f"{pair_str:>10}"
        )
    print(
        "\nunits: center/window in cm^-1, peaks in cm^-1, tau = -ln(I/I0) = peak * L_cm (dimensionless),"
        "\n       '*' after tau = outside [tau_min, tau_max] at this L_cm (informational, not filtered)."
        "\n       worst pair = tightest species pair in the window (every listed window already passed the"
        "\n       separation check -- that part IS a hard filter, so there's no pass/fail column for it)."
    )


def plot_combined_window_detail(result: CombinedSearchResult, index: int, margin_factor: float = 1.0):
    """Zoomed-in report for one CombinedWindowResult: every target species
    plotted separately, with the tightest pair and per-species metrics
    annotated.

    index: 0-based position into result.windows (already sorted best-first).
    """
    import matplotlib.pyplot as plt

    c = result.windows[index]
    width = c.hi - c.lo
    margin = margin_factor * width
    view_lo, view_hi = c.lo - margin, c.hi + margin
    mask = (result.nu_grid >= view_lo) & (result.nu_grid <= view_hi)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, alpha_i in result.species_spectra.items():
        ax.plot(result.nu_grid[mask], alpha_i[mask], label=name, lw=1.5)

    ax.axvspan(c.lo, c.hi, color="tab:green", alpha=0.12, label="selected window")
    for name, pos in c.line_position.items():
        if view_lo <= pos <= view_hi:
            ax.axvline(pos, color="0.6", ls=":", lw=0.8)

    names = list(c.peak_alpha)
    peaks_str = ", ".join(f"{n}={c.peak_alpha[n]:.2e}" for n in names)
    tau_str = ", ".join(f"{n}={c.tau[n]:.3f}{'' if c.tau_ok[n] else '*'}" for n in names)
    pair_str = f"{c.worst_pair[0]}/{c.worst_pair[1]}: {c.worst_pair[2]:.4f} (req {c.worst_pair[3]:.4f})" if c.worst_pair else "-"
    info = (
        f"peaks (cm$^{{-1}}$): {peaks_str}\n"
        f"tau (-ln(I/I0) at L_cm): {tau_str}\n"
        f"tightest pair: {pair_str}"
    )
    ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorption coefficient (cm$^{-1}$)")
    ax.set_title(f"Combined window #{index + 1} @ {c.center:.3f} cm$^{{-1}}$ -- {' + '.join(names)}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
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
