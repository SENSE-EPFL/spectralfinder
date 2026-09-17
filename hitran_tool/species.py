"""Molecule name <-> HITRAN id resolution and concentration-unit helpers.

Concentrations are handled internally as a plain mole fraction (0-1), i.e.
volume mixing ratio. Use the ppm_to_fraction/ppb_to_fraction/percent_to_fraction
helpers to convert from the units usually quoted in the literature (ppm, ppb, %).
"""
from __future__ import annotations

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


def molecule_id(name: str) -> int:
    try:
        return NAME_TO_ID[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown HITRAN molecule name {name!r}. Known names: {sorted(NAME_TO_ID)}"
        ) from exc


def ppm_to_fraction(ppm: float) -> float:
    return ppm * 1e-6


def ppb_to_fraction(ppb: float) -> float:
    return ppb * 1e-9


def percent_to_fraction(percent: float) -> float:
    return percent / 100.0
