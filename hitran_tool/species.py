"""Molecule name <-> HITRAN id resolution, isotopologue helpers, and
concentration-unit helpers.

Concentrations are handled internally as a plain mole fraction (0-1), i.e.
volume mixing ratio. Use the ppm_to_fraction/ppb_to_fraction/percent_to_fraction
helpers to convert from the units usually quoted in the literature (ppm, ppb, %).

Isotopologues: a mole fraction always refers to the WHOLE molecule (all
isotopologues together). HITRAN line intensities already include each
isotopologue's natural (terrestrial) abundance, so e.g. 13CH4 at "10 ppm CH4"
is simply CH4 iso_id=2 with mole_fraction=10e-6 -- its ~1.1% share comes from
HITRAN itself. A non-natural abundance (e.g. to simulate a different d13C)
can be passed explicitly where supported (TargetSpecies.abundance).
"""
from __future__ import annotations

import csv
from pathlib import Path

import hapi


def _build_name_to_id() -> dict:
    mapping = {}
    for molec_id in range(1, 60):
        try:
            name = hapi.moleculeName(molec_id)
        except Exception:
            continue
        if name:
            mapping[name] = molec_id
    return mapping


NAME_TO_ID = _build_name_to_id()
# Common spellings HAPI names differently.
_ALIASES = {"NO+": "NOp", "H3+": "H3p"}


def molecule_id(name: str) -> int:
    try:
        return NAME_TO_ID[_ALIASES.get(name, name)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown HITRAN molecule name {name!r}. Known names: {sorted(NAME_TO_ID)}"
        ) from exc


def isotopologue_ids(name: str) -> list:
    """Local (per-molecule) isotopologue ids HAPI knows for this molecule, e.g. [1, 2, 3, ...]."""
    m = molecule_id(name)
    return sorted(i for (mm, i) in hapi.ISO if mm == m)


def global_isotopologue_ids(name: str) -> list:
    """HITRANonline's global isotopologue ids for every isotopologue of this molecule
    (what hapi.fetch_by_ids expects)."""
    m = molecule_id(name)
    return [hapi.ISO[(m, i)][hapi.ISO_INDEX["id"]] for i in isotopologue_ids(name)]


def isotopologue_name(name: str, iso_id: int) -> str:
    """HITRAN isotopologue formula, e.g. ('CH4', 2) -> '(13C)H4'."""
    return hapi.ISO[(molecule_id(name), iso_id)][hapi.ISO_INDEX["iso_name"]]


def natural_abundance(name: str, iso_id: int) -> float:
    return hapi.ISO[(molecule_id(name), iso_id)][hapi.ISO_INDEX["abundance"]]


def species_label(name: str, iso_id: int) -> str:
    """Display name: the bare molecule name for the principal isotopologue
    (iso_id=1, the usual case), the HITRAN formula otherwise -- e.g. 'CH4',
    '(13C)H4', 'HD(16O)'."""
    return name if iso_id == 1 else isotopologue_name(name, iso_id)


def load_background(path: str | Path) -> dict:
    """Read a `molecule,mole_fraction` CSV (lines starting with '#' ignored) into
    {molecule: mole_fraction}. Every name is validated against HITRAN."""
    rows = [
        line for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    background = {}
    for row in csv.DictReader(rows):
        name = row["molecule"].strip()
        molecule_id(name)  # fail early on a typo
        background[name] = float(row["mole_fraction"])
    return background


def ppm_to_fraction(ppm: float) -> float:
    return ppm * 1e-6


def ppb_to_fraction(ppb: float) -> float:
    return ppb * 1e-9


def percent_to_fraction(percent: float) -> float:
    return percent / 100.0
