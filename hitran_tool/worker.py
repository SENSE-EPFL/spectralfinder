"""Entry point for running a search in a separate process (the web app's process pool).

Why a process and not a thread: HAPI keeps all its state in module globals
(hapi.LOCAL_TABLE_CACHE, and the `_{table}_slice` tables lines.sliced_table
writes there), and contextlib.redirect_stdout swaps sys.stdout for the whole
process -- two searches in threads of one process would overwrite each
other's tables. Each worker process has its own copy of all of that, and the
line lists it loads stay cached in it for the next search.
"""
from __future__ import annotations

import contextlib
import io

from .species import load_background
from .windows import CombinedSearchResult, SearchConfig, TargetSpecies, find_combined_windows


def search(targets: tuple, config: tuple, background_csv: str | None) -> CombinedSearchResult:
    """`targets` = ((name, mole_fraction, iso_id), ...), `config` = SearchConfig fields as
    (name, value) pairs -- plain picklable values, same as app.run_search's cache key."""
    background = load_background(background_csv) if background_csv else None
    with contextlib.redirect_stdout(io.StringIO()):  # HAPI and the fetches print a lot
        return find_combined_windows([TargetSpecies(n, x, iso) for n, x, iso in targets],
                                     SearchConfig(**dict(config)), background)
