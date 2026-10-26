"""SpectralFinder web interface: spectral-window search in the browser.

    uv run app.py

Same engine as run.py (hitran_tool.find_combined_windows). Deliberately simple:
only the essential parameters are on the page; the others keep their recommended
values (3-width separation, interference filter, line picked for its absorption,
no saturation filter) and are set in run.py. Outside 2000-9000 cm-1, lines are
downloaded first (several minutes), then cached in data/ -- except with
HITRAN_OFFLINE=1 (the public server, see Dockerfile): never a download, data/ only.

Each search runs in its own process (hitran_tool/worker.py), at most
MAX_PARALLEL_SEARCHES (default 2) at once for all visitors together.

A simulation can be saved (one JSON file, hitran_tool/saving.py) and reopened
in the page later without searching again.
"""
from __future__ import annotations

import base64
import dataclasses
import functools
import hashlib
import io
import math
import multiprocessing
import os
import tempfile
import threading
import time
import zipfile
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Scheme

from hitran_tool import SearchConfig, TargetSpecies, saving, worker
from hitran_tool.lines import retrieval_dates
from hitran_tool.report import conc, save_results_csv
from hitran_tool.species import NAME_TO_ID, isotopologue_ids, isotopologue_name, load_background

ROOT = Path(__file__).resolve().parent
UNITS = {"ppm": 1e-6, "ppb": 1e-9, "%": 1e-2}
# background gases span 1e-15 to 0.8: ppt needed too; each shown in the unit conc() would use
BACKGROUND_UNITS = {"%": (1e-2, 1e-3), "ppm": (1e-6, 1e-6), "ppb": (1e-9, 1e-9), "ppt": (1e-12, 0.0)}
STANDARD_BACKGROUND = list(load_background(ROOT / "atm_standard.csv").items())
# Validated categorical palette (dataviz skill, references/palette.md), in its fixed order: targets
# take the first slots, interferers the next ones. Dark mode has its own steps of the same hues.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
INK_LIGHT = dict(primary="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                 surface="#fcfcfb")
INK_DARK = dict(primary="#f0efec", secondary="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                surface="#1a1a19")
RED = {"color": "var(--critical)", "fontWeight": 600}  # out of range: the status color, never a series
# Table colors as props, not CSS: Dash's own table CSS outranks a stylesheet, and these inline
# styles still give way to style_data_conditional (red values, the selected row). Borders are
# in style.css: Dash merges a border prop badly with the per-column alignment.
TABLE_COLORS = dict(
    style_header={"backgroundColor": "var(--surface-2)", "color": "var(--text-2)", "fontWeight": 600},
    style_data={"backgroundColor": "var(--surface)", "color": "var(--text)"},
    style_cell={"textAlign": "left"},  # right-aligned, a dropdown's arrow covered the value
)
# Result tables: text left, numbers right (aligned digits, see style.css)
RESULT_TABLE = dict(TABLE_COLORS, style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "6px 12px", "fontFamily": "inherit",
                                "fontSize": "0.875rem"},
                    style_cell_conditional=[{"if": {"column_type": "numeric"}, "textAlign": "right"}])


@functools.cache
def search_pool() -> ProcessPoolExecutor:
    """Shared by every visitor: at most MAX_PARALLEL_SEARCHES searches run at once, the
    others wait their turn. Separate processes because HAPI's state is global (see
    hitran_tool/worker.py); 'spawn' because forking the server's threads isn't safe."""
    return ProcessPoolExecutor(max_workers=int(os.environ.get("MAX_PARALLEL_SEARCHES", "2")),
                               mp_context=multiprocessing.get_context("spawn"))


class ServerBusy(Exception):
    """Every search slot is taken."""


@functools.cache
def search_slots() -> threading.BoundedSemaphore:
    """Searches running or waiting, all visitors together; one more is refused."""
    return threading.BoundedSemaphore(int(os.environ.get("MAX_QUEUED_SEARCHES", "8")))


@functools.lru_cache(maxsize=8)
def run_search(targets: tuple, config: tuple, background: tuple | None):
    """Cached on its (hashable) inputs: changing a display option doesn't recompute.
    background: ((molecule, mole_fraction), ...) or None."""
    slots = search_slots()
    if not slots.acquire(blocking=False):
        raise ServerBusy
    try:
        future = search_pool().submit(worker.search, targets, config, background)
    except BaseException:
        slots.release()
        raise
    future.add_done_callback(lambda _: slots.release())
    return future.result()


# Opened simulations, like run_search's results: kept on the server (one process, see the
# Dockerfile), the browser only holds their key. Bounded: an evicted one is opened again.
_OPENED: OrderedDict = OrderedDict()
_OPENED_LOCK = threading.Lock()
_OPENED_MAX = 8
MAX_UPLOAD_MB = 20


def remember_opened(saved: tuple) -> str:
    """Key for a saving.loads() tuple, until _OPENED_MAX others were opened after it."""
    key = hashlib.sha256(saving.dumps(*saved).encode()).hexdigest()
    with _OPENED_LOCK:
        _OPENED[key] = saved
        _OPENED.move_to_end(key)
        while len(_OPENED) > _OPENED_MAX:
            _OPENED.popitem(last=False)
    return key


def load_run(run: dict) -> dict | None:
    """The run's targets, config, background, result, elapsed and source (file name if it
    was opened from a saved simulation); None if that one is no longer in memory."""
    if "opened" in run:
        with _OPENED_LOCK:
            saved = _OPENED.get(run["opened"])
        if saved is None:
            return None
        targets, config, background, result = saved
        return dict(targets=targets, config=config, background=background, result=result,
                    elapsed=None, source=run["name"])
    targets = [tuple(t) for t in run["targets"]]
    config = tuple(map(tuple, run["config"]))
    background = None if run["background"] is None else [tuple(b) for b in run["background"]]
    return dict(targets=targets, config=SearchConfig(**dict(config)), background=background,
                result=run_search(tuple(targets), config, None if background is None else tuple(background)),
                elapsed=run["elapsed"], source=None)


def describe_background(background: list | None, most: int = 8) -> str:
    """'standard background (42 gases)', or how it differs from the standard one."""
    if background is None:
        return "no background gases"
    standard, bg = dict(STANDARD_BACKGROUND), dict(background)
    changed = [m for m in bg if m not in standard or not math.isclose(bg[m], standard[m], rel_tol=1e-6)]
    removed = [m for m in standard if m not in bg]
    if not changed and not removed:
        return f"standard background ({len(bg)} gases)"
    parts = [f"{m} {conc(bg[m])}" for m in changed] + [f"no {m}" for m in removed]
    more = f" and {len(parts) - most} more" if len(parts) > most else ""
    return "background: standard except " + ", ".join(parts[:most]) + more


def describe(r: dict) -> str:
    """A run's parameters in one line -- for an opened one, the form doesn't show them."""
    c = r["config"]
    gases = ", ".join(f"{TargetSpecies(n, x, iso).label} {conc(x)}" for n, x, iso in r["targets"])
    return (f"{gases} · {c.sigma_min:g}–{c.sigma_max:g} cm⁻¹ · L {c.L_cm / 1e5:g} km · "
            f"{c.P * 1013.25:.3g} mbar · {c.T - 273.15:.3g} °C · scan {c.delta_sigma:g} cm⁻¹ · "
            f"absorption {100 * (1 - c.transmission_max):.3g}–{100 * (1 - c.transmission_min):.3g} % · "
            f"max. interference {100 * c.interference_max:.3g} % · {describe_background(r['background'])}")


def file_name(r: dict) -> str:
    labels = [TargetSpecies(n, x, iso).label for n, x, iso in r["targets"]]
    return f"simulation_{'_'.join(labels)}_{r['config'].sigma_min:g}-{r['config'].sigma_max:g}"


def csv_zip(result, background: list | None) -> bytes:
    """save_results_csv()'s files, plus the background gases used, zipped."""
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path in save_results_csv(result, tmp):
            z.write(path, os.path.basename(path))
        if background is not None:
            z.writestr("background.csv", "molecule,mole_fraction\n" + "".join(f"{m},{x!r}\n" for m, x in background))
    return buf.getvalue()


def background_rows(background: list) -> list:
    """Table rows for [(molecule, mole_fraction)], each in the most readable unit."""
    rows = []
    for mol, x in background:
        unit, (scale, _) = next((u, v) for u, v in BACKGROUND_UNITS.items() if x >= v[1])
        rows.append({"molecule": mol, "concentration": float(f"{x / scale:.4g}"), "unit": unit})
    return rows


def background_from_table(rows: list) -> tuple[list, list]:
    """([(molecule, mole_fraction)], errors)."""
    background, errors, seen = [], [], set()
    for row in rows:
        mol = row.get("molecule")
        if not mol:
            continue
        try:
            x = float(row.get("concentration")) * BACKGROUND_UNITS[row.get("unit") or "ppm"][0]
        except (TypeError, ValueError):
            x = 0.0
        if not 0 < x <= 1:
            errors.append(f"background {mol}: concentration missing, zero or above 100 %")
        elif mol in seen:
            errors.append(f"background {mol}: listed twice")
        else:
            seen.add(mol)
            background.append((mol, x))
    return background, errors


def targets_from_table(table: pd.DataFrame) -> tuple[list, list]:
    """(targets as (name, mole_fraction, iso_id), errors)."""
    targets, errors = [], []
    for _, row in table.iterrows():
        name = row.get("molecule")
        if not isinstance(name, str) or not name:
            continue
        conc = row.get("concentration")
        if pd.isna(conc) or conc <= 0:
            errors.append(f"{name}: concentration missing or zero")
            continue
        iso = 1 if pd.isna(row.get("isotope")) else int(row.get("isotope"))
        if iso not in isotopologue_ids(name):
            names = ", ".join(f"{k} = {isotopologue_name(name, k)}" for k in isotopologue_ids(name))
            errors.append(f"{name}: unknown isotope {iso} (possible: {names})")
            continue
        targets.append((name, float(conc) * UNITS[row.get("unit") or "ppm"], iso))
    if not targets and not errors:
        errors.append("add at least one gas to measure")
    return targets, errors


def number_input(key, label, value, min_value=None, max_value=None, step=None, format=None, help=None):
    return html.Div([html.Label(label, title=help, className="field-label"),
                     dcc.Input(id=key, type="number", value=value, min=min_value, max=max_value, step="any",
                               debounce=True)], className="field")


FIELDS = ["L_km", "delta_sigma", "P_mbar", "T_C", "sigma_min", "sigma_max", "min_abs", "max_abs",
          "interference_max", "with_background"]

# Asked by the HITRAN team: cite HITRAN and HAPI, give the data's retrieval date (HITRAN is
# updated continuously), and don't let this tool pass for official HITRAN software.
CITATIONS = [
    ("I. E. Gordon, L. S. Rothman, R. J. Hargreaves et al., The HITRAN2024 molecular spectroscopic database, "
     "J. Quant. Spectrosc. Radiat. Transfer 353, 109807 (2026)", "https://doi.org/10.1016/j.jqsrt.2026.109807"),
    ("R. V. Kochanov, I. E. Gordon, L. S. Rothman, P. Wcisło, C. Hill, J. S. Wilzewski, HITRAN Application "
     "Programming Interface (HAPI): A comprehensive approach to working with spectroscopic data, "
     "J. Quant. Spectrosc. Radiat. Transfer 177, 15–30 (2016)", "https://doi.org/10.1016/j.jqsrt.2016.03.005"),
]


def data_note() -> str:
    dates = retrieval_dates()
    when = "not downloaded yet" if dates is None else (
        f"retrieved on {dates[0]}" if dates[0] == dates[1] else f"retrieved between {dates[0]} and {dates[1]}")
    return (f"Line data: HITRANonline, {when}. HITRAN is updated continuously, also between editions, "
            "so results can change with the data version.")


def footer() -> html.Footer:
    return html.Footer([
        html.P(data_note()),
        html.P("Uses HITRAN data through HAPI. Please cite:"),
        html.Ul([html.Li([text, " ", html.A(url, href=url, target="_blank")]) for text, url in CITATIONS]),
        html.P("SpectralFinder is developed independently at EPFL. It is not developed, maintained or "
               "endorsed by the HITRAN team."),
    ], className="footer")

app = Dash(__name__, title="SpectralFinder")
server = app.server
# replaces Streamlit's /_stcore/health: used by the Dockerfile HEALTHCHECK and the Kubernetes probes
server.route("/health")(lambda: "ok")

app.layout = html.Div([
    html.Header([
        html.Div([html.H1("SpectralFinder"),
                  html.P("Spectral windows where several gases can be measured in one scan.", className="tagline")]),
        dcc.Checklist([" Dark mode"], [], id="dark", className="theme-switch"),
    ], className="page-head"),

    # ------------------------------------------------------------------ inputs ----

    html.Section([
        html.Div([
            html.Div([
                html.H3("Gases to measure"),
                dash_table.DataTable(
                    pd.DataFrame([
                        {"molecule": "CH4", "concentration": 10.0, "unit": "ppm", "isotope": 1},
                        {"molecule": "N2O", "concentration": 1.0, "unit": "ppm", "isotope": 1},
                        {"molecule": "H2O", "concentration": 30.0, "unit": "%", "isotope": 1},
                    ]).to_dict("records"),
                    editable=True, row_deletable=True, id="gases",
                    columns=[
                        {"name": "molecule", "id": "molecule", "presentation": "dropdown"},
                        {"name": "concentration", "id": "concentration", "type": "numeric"},
                        {"name": "unit", "id": "unit", "presentation": "dropdown"},
                        {"name": "isotope", "id": "isotope", "type": "numeric"},
                    ],
                    dropdown={
                        "molecule": {"options": [{"label": n, "value": n} for n in sorted(NAME_TO_ID)],
                                     "clearable": False},
                        "unit": {"options": [{"label": u, "value": u} for u in UNITS], "clearable": False},
                    },
                    tooltip_header={"isotope": "1 = main isotopologue; e.g. CH4: 2 = ¹³CH4, H2O: 4 = HDO"},
                    style_table={"overflowX": "auto"}, **TABLE_COLORS,
                ),
                html.Button("+ Add gas", id="add", className="ghost"),
            ], className="group"),

            html.Div([
                html.H3("Instrument"),
                html.Div([
                    number_input("L_km", "Effective L (km)", value=7.5, min_value=1e-5, format="%g",
                                 help="L = c·τ (ringdown)"),
                    number_input("delta_sigma", "Scan width (cm⁻¹)", value=0.55, min_value=0.01, step=0.05,
                                 format="%g"),
                    number_input("P_mbar", "Pressure (mbar)", value=30.0, min_value=0.01, format="%g"),
                    number_input("T_C", "Temperature (°C)", value=40.0, format="%g"),
                ], className="pair"),
            ], className="group"),

            html.Div([
                html.H3("Search"),
                html.Div([
                    number_input("sigma_min", "From (cm⁻¹)", value=4394.65, format="%g"),
                    number_input("sigma_max", "To (cm⁻¹)", value=4401.65, format="%g",
                                 help="the full 2000–9000 cm⁻¹ range: ~2 min"),
                    number_input("min_abs", "Min. absorption (%)", value=1.0, min_value=0.01, format="%g",
                                 help="detection limit: required SNR / spectrum SNR (10 / 1000)"),
                    number_input("max_abs", "Max. absorption (%)", value=20.0, min_value=0.1, max_value=99.0,
                                 format="%g"),
                    number_input("interference_max", "Max. interference (%)", value=5.0, min_value=0.1,
                                 format="%g"),
                    dcc.Checklist([{"label": " Background gases", "value": "on"}], ["on"], id="with_background",
                                  className="field check"),
                ], className="pair"),
            ], className="group"),
        ], className="columns"),

        html.Details([
            html.Summary("Edit background gases"),
            html.P("Interferers checked in every window, at these concentrations -- by default a standard "
                   "atmosphere (atm_standard.csv). A gas that is also measured takes the concentration given "
                   "above instead.", className="caption"),
            dash_table.DataTable(
                background_rows(STANDARD_BACKGROUND), id="background", editable=True, row_deletable=True,
                columns=[
                    {"name": "molecule", "id": "molecule", "presentation": "dropdown"},
                    {"name": "concentration", "id": "concentration", "type": "numeric"},
                    {"name": "unit", "id": "unit", "presentation": "dropdown"},
                ],
                dropdown={
                    "molecule": {"options": [{"label": n, "value": n} for n in sorted(NAME_TO_ID)],
                                 "clearable": False},
                    "unit": {"options": [{"label": u, "value": u} for u in BACKGROUND_UNITS], "clearable": False},
                },
                style_table={"maxHeight": "22rem", "overflowY": "auto"}, **TABLE_COLORS,
            ),
            html.Div([html.Button("+ Add gas", id="add-background", className="ghost"),
                      html.Button("Reset to standard", id="reset-background", className="ghost")],
                     className="actions"),
        ], className="background-box"),

        html.Div(id="errors"),
        html.Div([
            html.Button("Run search", id="go", className="primary"),
            dcc.Upload(html.Span("Open saved simulation…", className="secondary"), id="open",
                       accept=".json,application/json", max_size=MAX_UPLOAD_MB * 1024 * 1024, className="upload"),
        ], className="run-row"),
        html.Div(id="open-error"),
    ], className="card", **{"aria-label": "Parameters"}),
    dcc.Store(id="run"),

    # ------------------------------------------------------------------ results ----

    dcc.Loading([
        html.Div(id="search-error"),
        html.Section([
            html.Div([
                html.H2("Results"),
                html.Div([
                    html.Button("Save simulation", id="save", className="secondary",
                                title="parameters and results, to reopen here later"),
                    html.Button("Tables (CSV)", id="save-csv", className="secondary",
                                title="windows, interferers, temperature, other lines, gases to watch"),
                    dcc.Download(id="download"),
                ], className="actions"),
            ], className="section-head"),
            html.Div(id="summary"),
            html.Div([
                dash_table.DataTable(id="windows", data=[], columns=[], selected_rows=[0], **RESULT_TABLE),
                html.P("Click a window to see its spectrum. Absorption of each gas at its peak; "
                       "in red: outside the target range.", className="caption"),
            ], id="windows-box"),
            html.Div(id="detail"),
        ], id="results-section", className="card", hidden=True),
    ], type="dot", color="var(--accent)"),
    footer(),
], className="page")

# dark mode follows the system setting, like Streamlit did; the switch toggles it in the browser
app.clientside_callback("() => matchMedia('(prefers-color-scheme: dark)').matches ? [' Dark mode'] : []",
                        Output("dark", "value"), Input("dark", "id"))
app.clientside_callback("v => (document.documentElement.classList.toggle('dark', v.length > 0), 'theme-switch')",
                        Output("dark", "className"), Input("dark", "value"))


@app.callback(Output("gases", "data"), Input("add", "n_clicks"), State("gases", "data"), prevent_initial_call=True)
def add_gas(_, gases):
    return gases + [{"molecule": None, "concentration": None, "unit": "ppm", "isotope": 1}]


@app.callback(Output("background", "data"), Input("add-background", "n_clicks"),
              Input("reset-background", "n_clicks"), State("background", "data"), prevent_initial_call=True)
def edit_background(_, __, rows):
    if ctx.triggered_id == "reset-background":
        return background_rows(STANDARD_BACKGROUND)
    return rows + [{"molecule": None, "concentration": None, "unit": "ppm"}]


@app.callback(Output("with_background", "options"), Input("background", "data"))
def background_label(rows):
    """' Background gases (42)', '(43, edited)' once it differs from the standard list."""
    background, _ = background_from_table(rows)
    edited = describe_background(background).startswith("background: standard except")
    return [{"label": f" Background gases ({len(background)}{', edited' if edited else ''})", "value": "on"}]


def check(gases, background_table, L_km, delta_sigma, P_mbar, T_C, sigma_min, sigma_max, min_abs, max_abs,
          interference_max, with_background):
    """(targets, background or None if switched off, errors)."""
    targets, errors = targets_from_table(pd.DataFrame(gases))
    background = None
    if with_background:
        background, background_errors = background_from_table(background_table)
        errors += background_errors
    if None in (L_km, delta_sigma, P_mbar, T_C, sigma_min, sigma_max, min_abs, max_abs, interference_max):
        return targets, background, errors + ["a value is missing or invalid"]
    if min_abs >= max_abs:
        errors.append("min. absorption must be below max. absorption")
    if sigma_min >= sigma_max:
        errors.append("the spectral range is empty")
    return targets, background, errors


@app.callback(Output("errors", "children"), Output("go", "disabled"),
              Input("gases", "data"), Input("background", "data"), [Input(f, "value") for f in FIELDS])
def show_errors(*values):
    errors = check(*values)[2]
    return [html.Div(e, className="error") for e in errors], bool(errors)


@app.callback(Output("run", "data"), Output("search-error", "children"),
              Input("go", "n_clicks"), State("gases", "data"), State("background", "data"),
              [State(f, "value") for f in FIELDS], prevent_initial_call=True)
def search(_, gases, background_table, L_km, delta_sigma, P_mbar, T_C, sigma_min, sigma_max, min_abs, max_abs,
           interference_max, with_background):
    targets, background, errors = check(gases, background_table, L_km, delta_sigma, P_mbar, T_C, sigma_min,
                                        sigma_max, min_abs, max_abs, interference_max, with_background)
    if errors:
        return no_update, no_update
    config = SearchConfig(
        sigma_min=float(sigma_min), sigma_max=float(sigma_max), L_cm=float(L_km) * 1e5,
        delta_sigma=float(delta_sigma), T=float(T_C) + 273.15, P=float(P_mbar) / 1013.25,
        transmission_min=1 - max_abs / 100, transmission_max=1 - min_abs / 100,
        interference_max=float(interference_max) / 100,
    )
    t0 = time.time()
    try:
        run_search(tuple(targets), tuple(sorted(dataclasses.asdict(config).items())),
                   None if background is None else tuple(background))
    except ValueError as exc:
        return None, html.Div(f"Invalid parameters: {exc}", className="error")
    except ServerBusy:
        return None, html.Div("The server is busy with other searches: try again in a few minutes.",
                              className="error")
    except BrokenProcessPool:  # a worker died (e.g. out of memory): start a fresh pool
        search_pool.cache_clear()
        return None, html.Div("The search failed (server overloaded?): try again, or narrow the spectral range.",
                              className="error")
    return dict(targets=targets, config=sorted(dataclasses.asdict(config).items()),
                background=background, elapsed=time.time() - t0), None


@app.callback(Output("run", "data", allow_duplicate=True), Output("open-error", "children"),
              Output("open", "contents"),  # cleared, so opening the same file again triggers this
              Input("open", "contents"), State("open", "filename"), prevent_initial_call=True)
def open_simulation(contents, name):
    if contents is None:
        return no_update, no_update, no_update
    try:
        saved = saving.loads(base64.b64decode(contents.split(",", 1)[1]), STANDARD_BACKGROUND)
    except (ValueError, IndexError) as exc:  # binascii.Error is a ValueError
        return no_update, html.Div(f"Can't open {name}: {exc}", className="error"), None
    return dict(opened=remember_opened(saved), name=name), None, None


@app.callback(Output("download", "data"), Input("save", "n_clicks"), Input("save-csv", "n_clicks"),
              State("run", "data"), prevent_initial_call=True)
def download(_, __, run):
    r = load_run(run) if run else None
    if r is None:
        return no_update
    if ctx.triggered_id == "save":
        return dcc.send_string(saving.dumps(r["targets"], r["config"], r["background"], r["result"]),
                               f"{file_name(r)}.json", type="application/json")
    return dcc.send_bytes(csv_zip(r["result"], r["background"]), f"{file_name(r)}_csv.zip", type="application/zip")


def numeric(col: str, decimals: int) -> dict:
    return {"name": col, "id": col, "type": "numeric", "format": Format(precision=decimals, scheme=Scheme.fixed)}


@app.callback(Output("summary", "children"), Output("windows", "data"), Output("windows", "columns"),
              Output("windows", "style_data_conditional"), Output("windows", "selected_rows"),
              Output("windows-box", "hidden"), Output("detail", "children"), Output("results-section", "hidden"),
              # the clicked cell is cleared right away: the whole row is highlighted instead
              Output("windows", "active_cell"), Output("windows", "selected_cells"),
              Input("run", "data"), Input("windows", "active_cell"), Input("dark", "value"),
              State("windows", "selected_rows"))
def results(run, cell, dark, selected):
    return _results(run, cell, dark, selected) + (None, [])


def _results(run, cell, dark, selected) -> tuple:
    if run is None:
        return [], [], [], [], [0], True, [], True

    # the browser only keeps the search inputs (dcc.Store); the result comes back from run_search's
    # cache, or from the opened simulations
    r = load_run(run)
    if r is None:
        return ([html.Div("This simulation is no longer in the server's memory: open the file again.",
                          className="warning")], [], [], [], [0], True, [], False)
    result, config = r["result"], r["config"]
    labels = [TargetSpecies(n, x, iso).label for n, x, iso in r["targets"]]
    palette = SERIES_DARK if dark else SERIES_LIGHT
    color = {n: palette[i % len(palette)] for i, n in enumerate(labels)}

    summary = []
    if r["source"]:  # the form above still shows its own values, not the file's
        summary.append(html.Div([html.B(f"Opened from {r['source']}"), f": {describe(r)}"], className="info"))
    n, dropped = len(result.windows), result.n_rejected_interference
    stats = [html.Span([html.B(n), f" window{'s' * (n != 1)}"], className="stat")]
    if dropped:
        stats.append(html.Span([html.B(dropped), " dropped for interference"], className="stat"))
    if r["elapsed"] is not None and r["elapsed"] >= 1:
        stats.append(html.Span([html.B(f"{r['elapsed']:.0f} s")], className="stat"))
    summary.append(html.Div(stats, className="stats"))
    background_note = describe_background(r["background"])
    if not r["source"] and not background_note.startswith("standard"):  # the banner already says it
        summary.append(html.P(background_note[0].upper() + background_note[1:], className="caption"))
    if not result.windows:
        summary.append(html.Div("No window meets the criteria: try another spectral range or a higher "
                                "interference limit.", className="warning"))
        return summary, [], [], [], [0], True, [], False

    # a new search goes back to the first window; a click selects the clicked one
    if ctx.triggered_id == "run":
        choice = 0
    elif ctx.triggered_id == "windows" and cell:
        choice = cell["row"]
    else:
        choice = (selected or [0])[0]
    choice = min(choice, n - 1)

    rows, red = [], []
    for i, c in enumerate(result.windows):
        row = {"#": i + 1, "center (cm⁻¹)": c.center, "λ (nm)": 1e7 / c.center}
        for g in labels:
            row[f"{g} (%)"] = 100 * (1 - c.transmission[g])
            if not c.transmission_ok[g]:
                red.append({"if": {"row_index": i, "column_id": f"{g} (%)"}, **RED})
        row["max. interference (%)"] = 100 * max(c.interference.values())
        rows.append(row)
    columns = ([{"name": "#", "id": "#"}, numeric("center (cm⁻¹)", 3), numeric("λ (nm)", 2)]
               + [numeric(f"{g} (%)", 2) for g in labels] + [numeric("max. interference (%)", 2)])
    styles = [{"if": {"row_index": choice}, "backgroundColor": "var(--selected)", "fontWeight": 600}] + red

    return (summary, rows, columns, styles, [choice], False,
            window_detail(result, choice, labels, color, config, bool(dark)), False)


def window_detail(result, choice: int, labels: list, color: dict, config, dark: bool) -> list:
    c = result.windows[choice]
    spectra, others = result.window_spectra[choice], result.interferer_spectra[choice]
    ink = INK_DARK if dark else INK_LIGHT

    fig = go.Figure()
    # Interferers dashed, in the palette slots after the targets' (all grey, they couldn't be told apart)
    palette = SERIES_DARK if dark else SERIES_LIGHT
    for j, itf in enumerate([i for i in c.interferers if i.label in others][:3]):
        grid, alpha = others[itf.label]
        fig.add_trace(go.Scatter(x=grid, y=alpha, name=f"{itf.label} (interferer)",
                                 line=dict(color=palette[(len(labels) + j) % len(palette)], dash="dash", width=1.5),
                                 hovertemplate="α = %{y:.2e}"))
    for g, (grid, alpha) in spectra.items():
        fig.add_trace(go.Scatter(x=grid, y=alpha, name=g, line=dict(color=color[g], width=2),
                                 hovertemplate="α = %{y:.2e}"))
    # selected peaks: a ring in the gas's color, the label in ink (text never wears the series color)
    peaks = [(g, c.peak_position[g], c.peak_alpha[g], c.transmission[g], 14) for g in labels]
    peaks += [(p.species, p.position, p.alpha, p.transmission, 9) for p in c.extra_peaks]
    for g, pos, a, tr, size in peaks:
        main = size == 14
        fig.add_trace(go.Scatter(
            x=[pos], y=[a], mode="markers+text", showlegend=False, hoverinfo="skip", cliponaxis=False,
            marker=dict(symbol="circle-open", size=size, color=color[g], line=dict(width=2.5 if main else 2)),
            text=[f"<b>{g}</b> {100 * (1 - tr):.1f} %" if main else f"{100 * (1 - tr):.1f} %"],
            textposition="top center", textfont=dict(color=ink["primary"] if main else ink["secondary"],
                                                     size=12 if main else 11)))
    fig.update_yaxes(type="log", title="α (cm⁻¹)", exponentformat="power", dtick=1,  # decades only
                     gridcolor=ink["grid"], linecolor=ink["axis"], tickfont=dict(color=ink["muted"]))
    fig.update_xaxes(title="wavenumber (cm⁻¹)", tickformat=".2f", range=[c.lo, c.hi], showspikes=True,
                     spikemode="across", spikethickness=1, spikedash="solid", spikecolor=ink["muted"],
                     gridcolor=ink["grid"], linecolor=ink["axis"], tickfont=dict(color=ink["muted"]))
    fig.update_layout(height=440, margin=dict(t=48, b=50, l=60, r=48), hovermode="x unified",
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(color=ink["secondary"])),
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=ink["secondary"]),
                      template="plotly_dark" if dark else "plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=ink["surface"],
                      hoverlabel=dict(bgcolor=ink["surface"], font=dict(color=ink["primary"])))

    detail = [html.H3(f"Window #{choice + 1} · {c.center:.3f} cm⁻¹ · {1e7 / c.center:.2f} nm",
                      className="window-title"),
              dcc.Graph(figure=fig, config={"displaylogo": False, "responsive": True}, className="graph")]

    gas_rows, red = [], []
    for i, g in enumerate(labels):
        gas_rows.append({"gas": g, "peak (cm⁻¹)": c.peak_position[g], "absorption (%)": 100 * (1 - c.transmission[g]),
                         "interference (%)": 100 * c.interference.get(g, float("nan")),
                         "main interferer": c.main_interferer.get(g) or "–",
                         "T (%/K)": c.temperature_coeff.get(g, float("nan"))})
        if not c.transmission_ok[g]:
            red.append({"if": {"row_index": i, "column_id": "absorption (%)"}, **RED})
    for p in c.extra_peaks:  # in the absorption range by construction
        gas_rows.append({"gas": f"{p.species} (other line)", "peak (cm⁻¹)": p.position,
                         "absorption (%)": 100 * (1 - p.transmission), "interference (%)": 100 * p.interference,
                         "main interferer": "–", "T (%/K)": p.temperature_coeff})
    for i, row in enumerate(gas_rows):
        if row["interference (%)"] > 100 * config.interference_max:
            red.append({"if": {"row_index": i, "column_id": "interference (%)"}, **RED})
    detail.append(dash_table.DataTable(
        gas_rows, [{"name": "gas", "id": "gas"}, numeric("peak (cm⁻¹)", 4), numeric("absorption (%)", 2),
                   numeric("interference (%)", 2), {"name": "main interferer", "id": "main interferer"},
                   numeric("T (%/K)", 2)],
        style_data_conditional=red, **RESULT_TABLE))
    notes = []
    if c.extra_peaks:
        notes.append("Other line: another peak of the same gas in the window, also inside the absorption range "
                     "(small ring) -- listed for its temperature dependence.")
    notes += [f"Ratio {pair}: {k:+.3f} %/K → {1e3 * c.ratio_temperature_error[pair]:.2g} ‰ for "
              f"±{config.temperature_stability_K:g} K" for pair, k in c.ratio_temperature_coeff.items()]
    detail += [html.P(t, className="caption") for t in notes]

    if c.background_limits:
        shown = c.background_limits[:5]
        detail.append(html.H3("Background gases to watch"))
        detail.append(dash_table.DataTable(
            [{"gas": b.molecule, "standard": conc(b.mole_fraction), "limit": conc(b.max_mole_fraction),
              "× standard": times(b.factor), "disturbs": b.target} for b in shown],
            [{"name": col, "id": col} for col in ("gas", "standard", "limit", "× standard", "disturbs")],
            **dict(RESULT_TABLE, style_cell_conditional=[{"if": {"column_id": col}, "textAlign": "right"}
                                                         for col in ("standard", "limit", "× standard")]),
            style_data_conditional=[{"if": {"row_index": i, "column_id": "limit"}, **RED}
                                    for i, b in enumerate(shown) if b.factor < 10]))
        detail.append(html.P(f"Above its limit, the interference on the measured gas exceeds "
                             f"{100 * config.interference_max:g} % and the window fails; in red: "
                             "less than 10× the standard concentration.", className="caption"))
    return detail


def times(factor: float) -> str:
    """'×2,600' -- 2 significant digits, thousands separated, instead of '2.6e+03'."""
    return f"×{float(f'{factor:.2g}'):,.0f}" if factor >= 10 else f"×{factor:.2g}"


# A search worker (spawn) re-imports this file as __mp_main__ and must not start
# the server, only see the definitions above.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)
