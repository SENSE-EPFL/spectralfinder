"""Download the HITRAN lines of every molecule the web app offers, over the range it serves.

    uv run prefetch.py

The public app runs with HITRAN_OFFLINE=1 (never downloads, see hitran_tool/lines.py), so
data/ must already cover every molecule in its menu: the container runs this when it starts,
before the app (see Dockerfile). The image holds no HITRAN data, as the HITRAN team asked.
Molecules already cached over the range are skipped, so with data/ on a volume only the first
deployment downloads -- one HITRANonline request per molecule (59), which takes a while.
"""
from __future__ import annotations

import time

from hitran_tool.lines import ensure_lines
from hitran_tool.species import NAME_TO_ID

SIGMA_MIN, SIGMA_MAX = 2000.0, 9000.0  # cm^-1, the range the app serves

if __name__ == "__main__":
    names = sorted(NAME_TO_ID)
    for k, name in enumerate(names, 1):
        t0 = time.time()
        print(f"[{k}/{len(names)}] {name}", flush=True)
        ensure_lines(name, SIGMA_MIN, SIGMA_MAX)
        print(f"    {time.time() - t0:.0f} s", flush=True)
