"""Which change actually influences the selected windows?

Runs the validation case of run.py several times, adding this session's
changes ONE AT A TIME (cumulative), and compares the resulting ranked windows:

  V0  original program (git commit f6aaf57, before any of these changes)
  V1  peak read AT the stage-1 line (exact line center on the grid), instead
      of the max of the whole window
  V2  + line wings summed out to 5 cm^-1 (HAPI default: 50 half-widths)
  V3  + grid step 0.002 cm^-1 (was 0.01)
  V4  + interference check (background CSV + isotopologues) -- display only
  V5  + temperature coefficient -- display only
  V6  + saturation filter (total I/I0 >= 0.5 anywhere in the window)
  V7  + peak = top of the real curve (refined, measured width, taller top)
  V8  + distance-from-range penalty measured on tau, not on I/I0
  V9  + each gas's line/peak chosen for its absorption range, not the strongest
      = the current selection logic (on the original validation settings)

plus two side variants of V9, to see whether the saturation filter needs the
background gases at all:

  V9a saturation filter with the targets + their isotopologues only (no CSV)
  V9b saturation filter with the targets only

and the interference-in-the-ranking options (SearchConfig.rank_by_interference),
each compared with V9 (which doesn't rank on interference):

  RA  filter: drop windows where any target's interference > 5%
  RB  tiebreak: species in transmission range first, then lowest worst interference
  RC  score: distance from range + log10(1 + worst interference / 5%), no "in range" first

Earlier steps are reproduced on the CURRENT code by switching the later
features off (monkeypatching windows.py/spectrum.py internals), except V0,
which runs the original commit itself in a temporary git worktree.

    uv run compare_variants.py              # everything
    uv run compare_variants.py V9 RA RB RC  # only these variants

Prints a rank matrix (one row per window, one column per variant) and writes
results/variants.csv. Takes ~25 min (V4 onwards compute interferers).
"""
from __future__ import annotations

import contextlib
import csv
import dataclasses
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from math import exp, log10
from pathlib import Path

import run
from hitran_tool import find_combined_windows, spectrum, windows

ORIGINAL_COMMIT = "f6aaf57"
N_RANKED = 30  # windows ranked per variant -- deeper than 10 to locate the reference window
TOP = 10  # what's compared
REFERENCE = 4398.15  # real SubOcean CH4-N2O window, cm^-1
SAME_WINDOW = 0.05  # cm^-1: two variants' windows with closer centers are "the same window"

REPO = Path(__file__).resolve().parent

# ---------------------------------------------------------------- V0 --------

_V0_SNIPPET = """
import contextlib, io, json, run
from hitran_tool import find_combined_windows
run.config.max_windows = {n}
with contextlib.redirect_stdout(io.StringIO()):
    r = find_combined_windows(run.targets, run.config)
print(json.dumps([dict(center=c.center, lo=c.lo, hi=c.hi, transmission=c.transmission,
                       ok=c.transmission_ok) for c in r.windows]))
"""


def run_original() -> list:
    """The original commit, run as-is in a throwaway worktree sharing this repo's
    data/ (its HAPI .data tables are still there, so nothing is re-downloaded)."""
    tmp = Path(tempfile.mkdtemp(prefix="hitran_v0_"))
    wt = tmp / "wt"
    subprocess.run(["git", "worktree", "add", "--detach", str(wt), ORIGINAL_COMMIT],
                   cwd=REPO, check=True, capture_output=True)
    try:
        (wt / "data").symlink_to(REPO / "data")
        out = subprocess.run([sys.executable, "-c", _V0_SNIPPET.format(n=N_RANKED)], cwd=wt,
                             env=dict(os.environ, PYTHONPATH=str(wt)),
                             check=True, capture_output=True, text=True).stdout
        return json.loads(out.strip().splitlines()[-1])
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=REPO, capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------ V1..V9 --------

@contextlib.contextmanager
def patched(obj, **attrs):
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def _no_refine(grid, alpha, k):
    return float(grid[k]), float(alpha[k])


def _old_range_penalty(tau, tmin, tmax):
    """The penalty before V8: log10 distance measured on I/I0 instead of on tau."""
    tr = exp(-tau)
    if tr < tmin:
        return log10(tmin) - log10(max(tr, 1e-300))
    if tr > tmax:
        return log10(tr) - log10(tmax)
    return 0.0


def run_current(*, wings=True, fine_grid=True, interference=True, background=True,
                temperature=True, saturation=True, curve_top=True, penalty=True, range_lines=True,
                rank="none") -> list:
    config = dataclasses.replace(
        # frozen to the ORIGINAL validation settings (296 K, I/I0 in [0.80, 0.95]) so every
        # variant stays comparable with V0, whatever run.py now uses
        run.config, T=296.0, spectrum_snr=None, transmission_max=0.95,
        max_windows=N_RANKED, rank_by_interference=rank, pick_lines_for_range=range_lines,
        wavenumber_step=0.002 if fine_grid else 0.01,
        window_transmission_min=0.5 if saturation else 0.0,
    )
    patches = [
        # wing = 0 -> HAPI's own default cut (50 half-widths)
        patched(spectrum, SLICE_MARGIN=spectrum.SLICE_MARGIN if wings else 0.0),
        patched(windows,
                interferer_list=windows.interferer_list if interference else (lambda t, b: []),
                temperature_dependence=(windows.temperature_dependence if temperature
                                        else (lambda c, t, cfg: ({}, {}, {}))),
                # curve top off = read the curve AT the stage-1 line (it's on the grid; no
                # climbing to a nearby local max), stage-1 formula width, never switch to a
                # taller top -- so the stage-2 separation re-check sees stage 1's own numbers
                _climb=windows._climb if curve_top else (lambda alpha, k: k),
                _refine_top=windows._refine_top if curve_top else _no_refine,
                _measured_fwhm=windows._measured_fwhm if curve_top else (lambda *a: None),
                _WINDOW_MAX_TOLERANCE=windows._WINDOW_MAX_TOLERANCE if curve_top else float("inf"),
                _range_penalty=windows._range_penalty if penalty else _old_range_penalty),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        with contextlib.redirect_stdout(io.StringIO()):
            r = find_combined_windows(run.targets, config, run.background if background else None)
    return [dict(center=c.center, lo=c.lo, hi=c.hi, transmission=c.transmission, ok=c.transmission_ok,
                 interference=c.interference) for c in r.windows]


OFF = dict(wings=False, fine_grid=False, interference=False, temperature=False, saturation=False,
           curve_top=False, penalty=False, range_lines=False)
VARIANTS = [
    ("V0", "original program", None),
    ("V1", "peak read at the line", dict(OFF)),
    ("V2", "+ wings 5 cm-1", dict(OFF, wings=True)),
    ("V3", "+ grid 0.002", dict(OFF, wings=True, fine_grid=True)),
    ("V4", "+ interference (display)", dict(OFF, wings=True, fine_grid=True, interference=True)),
    ("V5", "+ T coeff (display)", dict(OFF, wings=True, fine_grid=True, interference=True, temperature=True)),
    ("V6", "+ saturation filter", dict(OFF, wings=True, fine_grid=True, interference=True, temperature=True,
                                       saturation=True)),
    ("V7", "+ curve top", dict(penalty=False, range_lines=False)),
    ("V8", "+ penalty on tau", dict(range_lines=False)),
    ("V9", "+ lines for the range = CURRENT", {}),
    ("V9a", "V9, saturation w/o CSV", dict(background=False)),
    ("V9b", "V9, saturation targets only", dict(interference=False)),
    ("RA", "rank: filter interf. > 5%", dict(rank="filter")),
    ("RB", "rank: tiebreak on interf.", dict(rank="tiebreak")),
    ("RC", "rank: score range + interf.", dict(rank="score")),
]
BASE = "V9"  # side variants are compared with the current program
COMPARE_WITH_BASE = ("V9a", "V9b", "RA", "RB", "RC")


# ---------------------------------------------------------- comparison ------

def _match(c, ws):
    """1-based rank of window c in ws, or None."""
    for i, w in enumerate(ws, 1):
        if abs(w["center"] - c) < SAME_WINDOW:
            return i
    return None


def _ref_rank(ws):
    for i, w in enumerate(ws, 1):
        if w["lo"] <= REFERENCE <= w["hi"]:
            return i
    return None


def _worst_interference(win) -> float | None:
    return max(win["interference"].values()) if win.get("interference") else None


def save_csv(path: Path, variants: list, results: dict) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "label", "rank", "center", "lo", "hi"]
                   + [f"I/I0 {t.label}" for t in run.targets] + ["all_in_range", "worst_interference"])
        for k, label, _ in variants:
            for j, win in enumerate(results.get(k, []), 1):
                w.writerow([k, label, j, win["center"], win["lo"], win["hi"]]
                           + [win["transmission"][t.label if t.label in win["transmission"] else t.name]
                              for t in run.targets]
                           + [all(win["ok"].values()), _worst_interference(win)])


def main() -> None:
    only = set(sys.argv[1:])
    variants = [v for v in VARIANTS if not only or v[0] in only]
    keys = [k for k, _, _ in variants]
    out = REPO / "results" / ("variants.csv" if not only else f"variants_{'_'.join(keys)}.csv")
    results, timing = {}, {}
    for key, label, opts in variants:
        t0 = time.time()
        print(f"running {key} ({label})...", flush=True)
        results[key] = run_original() if opts is None else run_current(**opts)
        timing[key] = time.time() - t0
        save_csv(out, variants, results)  # after EVERY variant: a stopped run keeps what it computed

    # every window that is top-10 in at least one variant, in order of first appearance
    rows = []
    for k in keys:
        for w in results[k][:TOP]:
            if all(abs(w["center"] - r) >= SAME_WINDOW for r in rows):
                rows.append(w["center"])
    rows.sort()

    print("\nRank of each window per variant ('-' = not in that variant's top "
          f"{N_RANKED}; ranks > {TOP} in parentheses)\n")
    print(f"{'window (cm-1)':>14} " + " ".join(f"{k:>5}" for k in keys))
    for c in rows:
        cells = []
        for k in keys:
            r = _match(c, results[k])
            cells.append("-" if r is None else (str(r) if r <= TOP else f"({r})"))
        print(f"{c:>14.3f} " + " ".join(f"{x:>5}" for x in cells))

    print(f"\nPer variant, compared with the PREVIOUS one in the list (V9a/V9b/R*: with {BASE}):\n")
    print(f"{'':4} {'change':<30} {'same top10':>10} {'same rank':>10} {'3/3 in top10':>13} "
          f"{'worst interf.':>13} {'4398.15 rank':>13} {'time':>7}")
    for i, (k, label, _) in enumerate(variants):
        prev = BASE if k in COMPARE_WITH_BASE and BASE in results else (keys[i - 1] if i else None)
        top = results[k][:TOP]
        n33 = sum(all(w["ok"].values()) for w in top)
        ref = _ref_rank(results[k])
        if prev:
            ptop = results[prev][:TOP]
            same = sum(_match(w["center"], ptop) is not None for w in top)
            same_rank = sum(_match(w["center"], ptop) == j for j, w in enumerate(top, 1))
            same_s, rank_s = f"{same}/{TOP}", f"{same_rank}/{TOP}"
        else:
            same_s = rank_s = "-"
        worst = [x for x in (_worst_interference(w) for w in top) if x is not None]
        worst_s = f"{100 * sorted(worst)[len(worst) // 2]:.1f}%" if worst else "-"  # median over top 10
        print(f"{k:<4} {label:<30} {same_s:>10} {rank_s:>10} {n33:>13} {worst_s:>13} "
              f"{(ref if ref else '>' + str(N_RANKED)):>13} {timing[k]:>6.0f}s")

    print("\nworst interf. = median over the top 10 of each window's worst target interference, as that "
          "variant computes it:\n                only the interferers it includes (V1-V3, V9b: other targets "
          "only; V9a: no background CSV).")
    print(f"\nSaved {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
