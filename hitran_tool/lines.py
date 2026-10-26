"""HITRAN line-list access (fetch-once, local cache) via HAPI.

One local table per MOLECULE, holding ALL of its isotopologues: targets pick
the isotopologue they need (iso_id filter in line_arrays/sliced_table), and
interferers use them all. Fetching per molecule rather than per isotopologue
keeps it to one HITRANonline request per molecule.

Storage: each table is cached as data/{table}.lines.npz (numeric columns +
HAPI header + fetched range + retrieval date), not as HAPI's own .data/.header
text files.
Why: hapi.db_begin() parses EVERY table in its folder on startup, at ~8 us per
line -- fine for 4 target gases, far too slow once ~40 background molecules
are cached too. The npz files load in milliseconds and only on demand; HAPI's
own folder (data/_fetch) is used only as a scratch area during a fetch and is
emptied right after.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from datetime import date
from pathlib import Path

import hapi
import numpy as np

from .constants import T_REF
from .species import NAME_TO_ID, _ALIASES, global_isotopologue_ids

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_FETCH_DIR = DATA_DIR / "_fetch"
# Server mode: never download, only use what's already cached (the public web app ships a
# prefilled data/ -- see prefetch.py); a range the cache doesn't cover is a ValueError.
OFFLINE = os.environ.get("HITRAN_OFFLINE", "") not in ("", "0")
_HAPI_STARTED = False
_FETCHED_RANGE = {}  # {table: (lo, hi)} -- coverage of what's loaded in hapi.LOCAL_TABLE_CACHE

# Columns HAPI's line-shape code needs; kept even when a fetch returns no lines,
# so an empty table still works everywhere a normal one does.
_REQUIRED_COLUMNS = {
    "molec_id": int, "local_iso_id": int, "nu": float, "sw": float, "a": float,
    "gamma_air": float, "gamma_self": float, "elower": float, "n_air": float,
    "delta_air": float, "gp": float, "gpp": float,
}


def _ensure_hapi() -> None:
    global _HAPI_STARTED
    if not _HAPI_STARTED:
        _FETCH_DIR.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            hapi.db_begin(str(_FETCH_DIR))
        _HAPI_STARTED = True


def _table_name(molecule: str) -> str:
    return _ALIASES.get(molecule, molecule)  # HAPI's own spelling, e.g. 'NO+' -> 'NOp'


def _npz_path(table: str) -> Path:
    return DATA_DIR / f"{table}.lines.npz"


def _register(table: str, data: dict, header: dict, lo: float, hi: float) -> None:
    header = dict(header, table_name=table, number_of_rows=int(len(data["nu"])))
    hapi.LOCAL_TABLE_CACHE[table] = {"header": header, "data": data, "filehandler": None}
    _FETCHED_RANGE[table] = (lo, hi)


def _load_npz(table: str) -> bool:
    path = _npz_path(table)
    if not path.exists():
        return False
    with np.load(path) as f:
        lo, hi = (float(x) for x in f["__range__"])
        header = json.loads(str(f["__header__"]))
        data = {k: f[k] for k in f.files if not k.startswith("__")}
    _register(table, data, header, lo, hi)
    return True


def _fetch(molecule: str, table: str, lo: float, hi: float) -> None:
    """Download every isotopologue of `molecule` in [lo, hi] and cache it as npz."""
    tmp = f"tmp_{table}"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            hapi.fetch_by_ids(tmp, global_isotopologue_ids(molecule), lo, hi)
        src = hapi.LOCAL_TABLE_CACHE[tmp]
        data = {
            k: np.ma.getdata(np.asarray(v)) for k, v in src["data"].items()
            if np.ma.getdata(np.asarray(v)).dtype.kind in "iufb"
        }
        header = copy.deepcopy(src["header"])
    except Exception as exc:
        # HITRANonline answers "Failed to retrieve data" when a molecule simply has no
        # line in [lo, hi] (e.g. SF6/CF4 outside their mid-IR bands) -- not an error here.
        if "Failed to retrieve data" not in str(exc):
            raise
        data = {k: np.empty(0, dtype=t) for k, t in _REQUIRED_COLUMNS.items()}
        header = {}
    finally:
        hapi.LOCAL_TABLE_CACHE.pop(tmp, None)
        for ext in (".data", ".header"):
            (_FETCH_DIR / f"{tmp}{ext}").unlink(missing_ok=True)

    np.savez(_npz_path(table), __range__=np.array([lo, hi]), __header__=np.array(json.dumps(header)),
             __retrieved__=np.array(date.today().isoformat()), **data)
    _register(table, data, header, lo, hi)


def retrieval_dates() -> tuple | None:
    """(oldest, newest) date the cached line lists were downloaded from HITRANonline, as
    'YYYY-MM-DD'; None if nothing is cached. HITRAN is updated continuously, also between
    editions, so results depend on it. Caches written before the date was stored use
    their file's date -- written right after the download."""
    dates = []
    for path in DATA_DIR.glob("*.lines.npz"):
        with np.load(path) as f:
            if "__retrieved__" in f.files:
                dates.append(str(f["__retrieved__"]))
            else:
                dates.append(date.fromtimestamp(path.stat().st_mtime).isoformat())
    return (min(dates), max(dates)) if dates else None


def ensure_lines(molecule: str, sigma_min: float, sigma_max: float) -> str:
    """Return the local table name holding ALL isotopologues of `molecule`,
    fetching from HITRANonline if the cache doesn't already cover
    [sigma_min, sigma_max] (then re-fetching the union of the old and new range).

    Coverage is tracked explicitly (stored with the table) rather than inferred
    from the actual line positions: real spectral lines are at discrete, sparse
    positions, so nu.min()/nu.max() almost never land exactly on the requested
    boundary -- comparing against them made this re-fetch on every call.
    """
    if _ALIASES.get(molecule, molecule) not in NAME_TO_ID:
        raise ValueError(f"Unknown HITRAN molecule name {molecule!r}")
    _ensure_hapi()
    table = _table_name(molecule)
    if table not in _FETCHED_RANGE:
        _load_npz(table)

    lo_hi = _FETCHED_RANGE.get(table)
    if lo_hi is None or lo_hi[0] > sigma_min or lo_hi[1] < sigma_max:
        if OFFLINE:
            have = f"{lo_hi[0]:g}-{lo_hi[1]:g} cm^-1" if lo_hi else "none"
            raise ValueError(f"no {molecule} lines cached for {sigma_min:g}-{sigma_max:g} cm^-1 "
                             f"(cached: {have}) and downloads are off (HITRAN_OFFLINE)")
        new_lo = min(sigma_min, lo_hi[0]) if lo_hi else sigma_min
        new_hi = max(sigma_max, lo_hi[1]) if lo_hi else sigma_max
        print(f"  fetching {molecule} lines (all isotopologues) {new_lo:g}-{new_hi:g} cm^-1 from HITRANonline...")
        _fetch(molecule, table, new_lo, new_hi)
    return table


SLICE_MARGIN = 5.0  # cm^-1, generous vs. typical line FWHM (~1e-3 to a few 1e-1 cm^-1 in this project)


def sliced_table(table: str, sigma_min: float, sigma_max: float, iso_id: int | None = None,
                 margin: float = SLICE_MARGIN) -> str:
    """Return the name of a table containing only the rows of `table` within
    [sigma_min - margin, sigma_max + margin] (and, if given, only isotopologue
    `iso_id`), built via a vectorized numpy mask (no fetch, no re-parsing --
    `table` must already be loaded).

    Why this exists: HAPI's own absorptionCoefficient_* iterates every row of
    the SourceTable in a pure-Python loop regardless of the requested
    WavenumberRange or Components -- a table covering 2000-9000 cm^-1 and all
    isotopologues makes even a 0.5 cm^-1 query pay for ~100k+ rows. Slicing
    first with numpy is orders of magnitude cheaper, and the destination
    table it produces is what should be passed to absorptionCoefficient_*.
    """
    dest = f"_{table}_slice"
    src = hapi.LOCAL_TABLE_CACHE[table]
    nu = np.asarray(src["data"]["nu"])
    mask = (nu >= sigma_min - margin) & (nu <= sigma_max + margin)
    if iso_id is not None:
        mask &= np.asarray(src["data"]["local_iso_id"]) == iso_id
    new_data = {k: np.asarray(v)[mask] for k, v in src["data"].items()}
    new_header = copy.deepcopy(src["header"])
    new_header["table_name"] = dest
    new_header["number_of_rows"] = int(mask.sum())
    hapi.LOCAL_TABLE_CACHE[dest] = {"header": new_header, "data": new_data, "filehandler": None}
    return dest


def isotopologues_present(table: str, sigma_min: float, sigma_max: float,
                          margin: float = SLICE_MARGIN) -> list:
    """Local iso ids that have at least one line in [sigma_min - margin, sigma_max + margin]."""
    d = hapi.LOCAL_TABLE_CACHE[table]["data"]
    nu = np.asarray(d["nu"])
    mask = (nu >= sigma_min - margin) & (nu <= sigma_max + margin)
    return sorted(int(i) for i in np.unique(np.asarray(d["local_iso_id"])[mask]))


def line_arrays(table: str, sigma_min: float, sigma_max: float, T: float,
                iso_id: int | None = None) -> dict:
    """
    Read the lines of `table` within [sigma_min, sigma_max] (only isotopologue
    `iso_id` if given) and return them as a dict of arrays (same order/length
    for every key):
    - 'nu': wavenumber (cm^-1), unshifted (vacuum, zero pressure)
    - 'sw': line intensity at T (cm^-1/(molecule cm^-2)) -- rescaled, see below.
      Includes the isotopologue's natural abundance (HITRAN convention).
    - 'gamma_air': air-broadened half-width at half-maximum (cm^-1/atm, at 296K/1atm)
    - 'gamma_self': self-broadened half-width at half-maximum (cm^-1/atm, at 296K/1atm)
    - 'n_air': temperature exponent for the air-broadened half-width
    - 'delta_air': air pressure shift of the line center (cm^-1/atm)
    - 'elower': lower-state energy of the transition (cm^-1)
    - 'molec_id': HITRAN molecule ID
    - 'iso_id': isotopologue ID

    'sw' is HITRAN's reference intensity at 296 K; it is rescaled to T with
    HAPI's EnvironmentDependency_Intensity, which implements:

        sw(T) = sw(296K) * ratio_Q * ratio_boltzmann * ratio_stim_emission

        ratio_Q             = Q(296K) / Q(T)
        ratio_boltzmann     = exp(-c2*elower/T)      / exp(-c2*elower/296K)
        ratio_stim_emission = (1 - exp(-c2*nu/T))    / (1 - exp(-c2*nu/296K))

    ratio_boltzmann: how the lower-state population changes with T.
    ratio_stim_emission: stimulated-emission correction.
    ratio_Q: partition-function normalization (computed per isotopologue
    below, since Q is isotopologue-specific).
    c2 = hc/kB = 1.4388 cm*K.
    """
    d = hapi.LOCAL_TABLE_CACHE[table]["data"]
    nu = np.asarray(d["nu"], dtype=float)
    mask = (nu >= sigma_min) & (nu <= sigma_max)
    if iso_id is not None:
        mask &= np.asarray(d["local_iso_id"]) == iso_id

    def col(name, dtype=float):
        return np.asarray(d[name], dtype=dtype)[mask]

    nu = nu[mask]
    sw_ref = col("sw")
    elower = col("elower")
    molec_id = col("molec_id", int)
    iso = col("local_iso_id", int)

    sw_T = np.empty_like(sw_ref)
    for m in np.unique(molec_id):
        for i in np.unique(iso[molec_id == m]):
            sel = (molec_id == m) & (iso == i)
            sigma_t = hapi.partitionSum(int(m), int(i), T)
            sigma_tref = hapi.partitionSum(int(m), int(i), T_REF)
            sw_T[sel] = hapi.EnvironmentDependency_Intensity(
                sw_ref[sel], T, T_REF, sigma_t, sigma_tref, elower[sel], nu[sel]
            )

    return dict(
        nu=nu, sw=sw_T, gamma_air=col("gamma_air"), gamma_self=col("gamma_self"),
        n_air=col("n_air"), delta_air=col("delta_air"), elower=elower,
        molec_id=molec_id, iso_id=iso,
    )
