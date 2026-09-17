"""HITRAN line-list access (fetch-once, local cache) via HAPI."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import hapi
import numpy as np

from .constants import T_REF
from .species import molecule_id

DATA_DIR = Path(__file__).resolve().parent.parent / "data" # search for the path to the data directory relative to this file
_DB_STARTED = False


def _ensure_db() -> None:
    global _DB_STARTED
    if not _DB_STARTED: 
        DATA_DIR.mkdir(exist_ok=True) # create the data directory if it doesn't exist
        hapi.db_begin(str(DATA_DIR)) # start the HAPI database in the data directory
        _DB_STARTED = True


def _range_file(table: str) -> Path:
    return DATA_DIR / f"{table}.fetched_range.json"


def ensure_lines(molecule: str, sigma_min: float, sigma_max: float, iso_id: int = 1) -> str:
    """Return the local HAPI table name for `molecule`, fetching from HITRANonline
    if the cached table doesn't already cover [sigma_min, sigma_max].

    Coverage is tracked in a sidecar {table}.fetched_range.json rather than
    inferred from the actual line positions: real spectral lines are at
    discrete, sparse positions, so nu.min()/nu.max() of a just-fetched table
    almost never land exactly on the requested boundary -- comparing against
    them made ensure_lines re-fetch on every single call.
    """
    _ensure_db()
    table = molecule  # the name of the molecule ex : "H2O" or "CO2"
    m_id = molecule_id(molecule)  # we get the molecule id from the molecule name

    range_file = _range_file(table)
    fetched_lo, fetched_hi = (None, None)
    if range_file.exists():
        fetched_lo, fetched_hi = json.loads(range_file.read_text())

    covers = (
        table in hapi.LOCAL_TABLE_CACHE
        and fetched_lo is not None
        and fetched_lo <= sigma_min
        and fetched_hi >= sigma_max
    )

    if not covers:  # fetch the union of what's already covered and what's newly requested
        new_lo = min(sigma_min, fetched_lo) if fetched_lo is not None else sigma_min
        new_hi = max(sigma_max, fetched_hi) if fetched_hi is not None else sigma_max
        hapi.fetch(table, m_id, iso_id, new_lo, new_hi)
        range_file.write_text(json.dumps([new_lo, new_hi]))

    return table


_SLICE_MARGIN = 5.0  # cm^-1, generous vs. typical line FWHM (~1e-3 to a few 1e-1 cm^-1 in this project)


def sliced_table(table: str, sigma_min: float, sigma_max: float, margin: float = _SLICE_MARGIN) -> str:
    """Return the name of a table containing only the rows of `table` within
    [sigma_min - margin, sigma_max + margin], built via a vectorized numpy
    mask (no fetch, no re-parsing -- `table` must already be loaded).

    Why this exists: as a table's cached coverage grows (e.g. a molecule
    first queried over a wide range, then reused for a narrow one), HAPI's
    own absorptionCoefficient_* iterates every row of the SourceTable in a
    pure-Python loop regardless of the requested WavenumberRange -- a table
    grown to cover 2000-9000 cm^-1 makes even a 100 cm^-1 query pay for all
    ~200k rows. Slicing first with numpy (this function) avoids that: it's
    a single vectorized mask, orders of magnitude cheaper than HAPI's
    per-row loop, and the destination table it produces is what should be
    passed to absorptionCoefficient_* instead of the raw `table`.
    """
    dest = f"_{table}_slice"
    src = hapi.LOCAL_TABLE_CACHE[table]
    nu = np.asarray(src["data"]["nu"])
    mask = (nu >= sigma_min - margin) & (nu <= sigma_max + margin)
    new_data = {k: np.asarray(v)[mask] for k, v in src["data"].items()}
    new_header = copy.deepcopy(src["header"])
    new_header["table_name"] = dest
    new_header["number_of_rows"] = int(mask.sum())
    hapi.LOCAL_TABLE_CACHE[dest] = {"header": new_header, "data": new_data, "filehandler": None}
    return dest


def line_arrays(table: str, sigma_min: float, sigma_max: float, T: float) -> dict:
    """
    Read the lines of `table` within [sigma_min, sigma_max] and return them as
    a dict of arrays (same order/length for every key):
    - 'nu': wavenumber (cm^-1)
    - 'sw': line intensity at T (cm^-1/(molecule cm^-2)) -- rescaled, see below
    - 'gamma_air': air-broadened half-width at half-maximum (cm^-1/atm, at 296K/1atm)
    - 'gamma_self': self-broadened half-width at half-maximum (cm^-1/atm, at 296K/1atm)
    - 'n_air': temperature exponent for the air-broadened half-width
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
    nu = np.asarray(hapi.getColumn(table, "nu"), dtype=float)
    mask = (nu >= sigma_min) & (nu <= sigma_max)

    nu = nu[mask]
    sw_ref = np.asarray(hapi.getColumn(table, "sw"), dtype=float)[mask]
    gamma_air = np.asarray(hapi.getColumn(table, "gamma_air"), dtype=float)[mask]
    gamma_self = np.asarray(hapi.getColumn(table, "gamma_self"), dtype=float)[mask]
    n_air = np.asarray(hapi.getColumn(table, "n_air"), dtype=float)[mask]
    elower = np.asarray(hapi.getColumn(table, "elower"), dtype=float)[mask]
    molec_id = np.asarray(hapi.getColumn(table, "molec_id"), dtype=int)[mask]
    iso_id = np.asarray(hapi.getColumn(table, "local_iso_id"), dtype=int)[mask]

    sw_T = np.empty_like(sw_ref)
    for m in np.unique(molec_id):
        for i in np.unique(iso_id[molec_id == m]):
            sel = (molec_id == m) & (iso_id == i)
            sigma_t = hapi.partitionSum(int(m), int(i), T)
            sigma_tref = hapi.partitionSum(int(m), int(i), T_REF)
            sw_T[sel] = hapi.EnvironmentDependency_Intensity(
                sw_ref[sel], T, T_REF, sigma_t, sigma_tref, elower[sel], nu[sel]
            )

    return dict(
        nu=nu, sw=sw_T, gamma_air=gamma_air, gamma_self=gamma_self,
        n_air=n_air, elower=elower, molec_id=molec_id, iso_id=iso_id,
    )
