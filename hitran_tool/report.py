"""Text report and quick-look plot for a SearchResult."""
from __future__ import annotations

from .windows import CombinedWindowResult, SearchResult


def print_report(result: SearchResult, target_name: str) -> None:
    if not result.windows:
        print(f"No candidate window found for {target_name} with the current settings.")
        return

    print(f"Top {len(result.windows)} candidate window(s) for {target_name}\n")
    header = (
        f"{'#':>2} {'center':>10} {'window':>17} {'peak a':>10} {'interf.':>9} "
        f"{'sep. ok':>8} {'L @ tau':>10} {'dT%/10K':>8} {'dP%/0.1atm':>10}  dominant interferer"
    )
    print(header)
    print("-" * len(header))
    for i, w in enumerate(result.windows, 1):
        L_str = f"{w.suggested_path_length_cm:.2e}cm" if w.suggested_path_length_cm != float("inf") else "inf"
        print(
            f"{i:>2} {w.center:>10.3f} [{w.lo:>7.3f},{w.hi:>7.3f}] {w.target_peak_alpha:>10.3e} "
            f"{w.interference_index:>9.3f} {str(w.separation_ok):>8} {L_str:>10} "
            f"{w.sensitivity_T_percent_per_10K:>8.2f} {w.sensitivity_P_percent_per_0p1atm:>10.2f}  "
            f"{w.dominant_interferer or '-'}"
        )
    print(
        "\nunits: center/window in cm^-1, peak alpha in cm^-1, interf. = interferer/target peak alpha ratio,"
        "\n       L @ tau = path length (cm) giving optical depth tau_target, dT/dP = %% change in peak alpha."
    )


def print_report_multi(results: dict) -> None:
    """results: {target_name: SearchResult}, as returned by find_windows_multi()."""
    for target_name, result in results.items():
        print_report(result, target_name)
        print()


def print_combined_report(windows: list, names: list | None = None) -> None:
    """windows: list of CombinedWindowResult, as returned by find_combined_windows()."""
    if not windows:
        print("No window found where all targets are simultaneously detectable.")
        return

    names = names or list(windows[0].peak_alpha)
    print(f"Top {len(windows)} combined window(s) for {' + '.join(names)}\n")
    header = (
        f"{'#':>2} {'center':>10} {'window':>17} "
        + " ".join(f"{n + ' peak':>14}" for n in names)
        + f" {'worst pair':>10} {'sep ok':>7} {'L (cm)':>10}  bg. dominant"
    )
    print(header)
    print("-" * len(header))
    for i, c in enumerate(windows, 1):
        peaks = " ".join(f"{c.peak_alpha[n]:>14.3e}" for n in names)
        pair_str = f"{c.worst_pair[0][:2]}/{c.worst_pair[1][:2]}" if c.worst_pair else "-"
        L_str = f"{c.suggested_path_length_cm_combined:.2e}" if c.suggested_path_length_cm_combined != float("inf") else "inf"
        print(
            f"{i:>2} {c.center:>10.3f} [{c.lo:>7.3f},{c.hi:>7.3f}] {peaks} "
            f"{pair_str:>10} {str(c.separation_ok):>7} {L_str:>10}  {c.dominant_background or '-'}"
        )
    print(
        "\nunits: center/window in cm^-1, peaks in cm^-1, worst pair = tightest species pair in the window,"
        "\n       L (cm) = path length covering the weakest species at tau_target (max over species)."
    )


def plot_windows(result: SearchResult, target_name: str, top_n: int = 5):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.nu_grid, result.alpha_target, label=target_name, lw=1)
    if result.interferer_spectra:
        ax.plot(result.nu_grid, result.alpha_interference_total, label="interferers (sum)", lw=1, alpha=0.7)

    for i, w in enumerate(result.windows[:top_n], 1):
        ax.axvspan(w.lo, w.hi, color="tab:green", alpha=0.15)
        ax.text(w.center, ax.get_ylim()[1], str(i), ha="center", va="bottom")

    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorption coefficient (cm$^{-1}$)")
    ax.set_title(f"Candidate detection windows for {target_name}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_window_detail(result: SearchResult, target_name: str, index: int, margin_factor: float = 1.0):
    """Zoomed-in, one-window report: target vs. EACH interferer plotted
    separately (not summed, unlike plot_windows), with the window bounds,
    the nearest interfering line, and the window's metrics annotated.

    index: 0-based position into result.windows (already sorted best-first).
    margin_factor: how much context to show around [lo, hi], in units of the
    window width (1.0 = show one extra window-width on each side).
    """
    import matplotlib.pyplot as plt

    w = result.windows[index]
    width = w.hi - w.lo
    margin = margin_factor * width
    view_lo, view_hi = w.lo - margin, w.hi + margin
    mask = (result.nu_grid >= view_lo) & (result.nu_grid <= view_hi)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.nu_grid[mask], result.alpha_target[mask], label=target_name, lw=1.5, color="tab:blue")
    for name, alpha_i in result.interferer_spectra.items():
        ax.plot(result.nu_grid[mask], alpha_i[mask], label=name, lw=1, ls="--", alpha=0.8)

    ax.axvspan(w.lo, w.hi, color="tab:green", alpha=0.12, label="selected window")
    if w.nearest_interferer_line is not None and view_lo <= w.nearest_interferer_line <= view_hi:
        ax.axvline(w.nearest_interferer_line, color="tab:red", ls=":", lw=1,
                   label=f"nearest line ({w.dominant_interferer or '?'})")

    L_str = f"{w.suggested_path_length_cm:.2e} cm" if w.suggested_path_length_cm != float("inf") else "inf"
    separation_str = f"{w.separation:.4f}" if w.separation is not None else "-"
    info = (
        f"peak alpha = {w.target_peak_alpha:.3e} cm$^{{-1}}$\n"
        f"interference index = {w.interference_index:.3f}  (dominant: {w.dominant_interferer or '-'})\n"
        f"separation = {separation_str} cm$^{{-1}}$ (required {w.required_separation:.4f}), ok={w.separation_ok}\n"
        f"suggested path length @ tau = {L_str}\n"
        f"sensitivity: {w.sensitivity_T_percent_per_10K:+.2f}%/10K, "
        f"{w.sensitivity_P_percent_per_0p1atm:+.2f}%/0.1atm"
    )
    ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorption coefficient (cm$^{-1}$)")
    ax.set_title(f"{target_name} -- window #{index + 1} @ {w.center:.3f} cm$^{{-1}}$")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def save_window_details(result: SearchResult, target_name: str, output_dir: str, **kwargs) -> list:
    """Save one PNG per window (plot_window_detail) into output_dir. Returns the file paths."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i in range(len(result.windows)):
        fig = plot_window_detail(result, target_name, i, **kwargs)
        path = os.path.join(output_dir, f"{target_name}_window_{i + 1:02d}.png")
        fig.savefig(path, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        paths.append(path)
    return paths
