"""Save / reopen a whole simulation -- its inputs and every result, spectra
included -- as one JSON file (the web app's save / open buttons).

JSON rather than pickle: on the public server the file to reopen is uploaded
by a visitor, and unpickling an untrusted file runs whatever code it holds.
Reopening rebuilds the result objects as they were, so the report, plots and
CSV export work on them without searching again.

Spectra are most of the size (a ~300-point grid per window, for every target
and every interferer isotopologue): the grid is stored once per window and
alpha with 6 significant digits -- plenty for a plot, ~1-2 MB for 10 windows.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np

from .interference import BackgroundLimit, InterfererResult
from .windows import CombinedSearchResult, CombinedWindowResult, ExtraPeak, SearchConfig

FORMAT = "hitran-window-search"
# 1: background was only on/off (the standard list); 2: the background list itself
VERSION = 2


def dumps(targets: list, config: SearchConfig, background: list | None, result: CombinedSearchResult) -> str:
    """targets: [(name, mole_fraction, iso_id)], background: [(molecule, mole_fraction)] or
    None, as the web app holds them."""
    windows = []
    for c, spectra, others in zip(result.windows, result.window_spectra, result.interferer_spectra):
        grid = next(iter(spectra.values()))[0]
        windows.append(dict(
            result=dataclasses.asdict(c),
            grid=grid.tolist(),
            spectra={n: _rounded(a) for n, (_, a) in spectra.items()},
            interferer_spectra={n: _rounded(a) for n, (_, a) in others.items()},
        ))
    summary = {f.name: getattr(result, f.name) for f in dataclasses.fields(result)
               if f.name not in ("windows", "window_spectra", "interferer_spectra")}
    return json.dumps(dict(format=FORMAT, version=VERSION, targets=[list(t) for t in targets],
                           config=dataclasses.asdict(config),
                           background=None if background is None else [list(b) for b in background],
                           summary=summary, windows=windows), default=_numpy_scalar)


def loads(text: str | bytes, standard_background: list | None = None) -> tuple:
    """(targets, config, background, result), as passed to dumps(). ValueError if the file
    isn't a saved simulation of a version this code can read. standard_background: what
    a version-1 file's 'background on' meant."""
    try:
        d = json.loads(text)
        if not isinstance(d, dict) or d.get("format") != FORMAT:
            raise ValueError("not a saved simulation")
        if d.get("version") == 1:
            background = list(standard_background or []) if d["with_background"] else None
        elif d.get("version") == VERSION:
            background = None if d["background"] is None else [(str(m), float(x)) for m, x in d["background"]]
        else:
            raise ValueError(f"saved with format version {d.get('version')}, this version reads 1-{VERSION}")
        targets = [(str(n), float(x), int(iso)) for n, x, iso in d["targets"]]
        config = SearchConfig(**d["config"])
        windows, window_spectra, interferer_spectra = [], [], []
        for w in d["windows"]:
            grid = np.asarray(w["grid"], dtype=float)
            windows.append(_window(w["result"]))
            window_spectra.append({n: (grid, np.asarray(a, dtype=float)) for n, a in w["spectra"].items()})
            interferer_spectra.append({n: (grid, np.asarray(a, dtype=float))
                                       for n, a in w["interferer_spectra"].items()})
        summary = dict(d["summary"], transmission_range=tuple(d["summary"]["transmission_range"]))
        result = CombinedSearchResult(windows, window_spectra, interferer_spectra, **summary)
        return targets, config, background, result
    except (KeyError, TypeError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
        raise ValueError(f"can't read this file: {exc}") from exc


def _numpy_scalar(x):
    """json.dumps fallback: a numpy bool/int (np.float64 already is a float) as its Python value."""
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(f"can't save a {type(x).__name__}")


def _rounded(alpha: np.ndarray) -> list:
    return [float(f"{v:.6g}") for v in alpha]


def _window(d: dict) -> CombinedWindowResult:
    """CombinedWindowResult from dataclasses.asdict() after a JSON round trip: nested
    dataclasses came back as dicts, tuples as lists."""
    d = dict(d)
    d["worst_pair"] = tuple(d["worst_pair"]) if d["worst_pair"] else None
    d["temperature_sweep"] = {n: [tuple(p) for p in s] for n, s in d["temperature_sweep"].items()}
    d["interferers"] = [InterfererResult(**r) for r in d["interferers"]]
    d["extra_peaks"] = [ExtraPeak(**p) for p in d["extra_peaks"]]
    d["background_limits"] = [BackgroundLimit(**b) for b in d["background_limits"]]
    return CombinedWindowResult(**d)
