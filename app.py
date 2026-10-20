"""Interface graphique pour tester la sélection de fenêtres spectrales.

    uv run app.py

Même moteur que run.py (hitran_tool.find_combined_windows). Volontairement
simple : seuls les paramètres essentiels sont dans la page ; les autres gardent
les valeurs recommandées (séparation 3 largeurs, filtre d'interférence, raie
choisie pour son absorption, pas de filtre de saturation) et se règlent dans
run.py. Hors 2000-9000 cm-1, les raies sont d'abord téléchargées (plusieurs
minutes), ensuite elles sont en cache dans data/ -- sauf avec HITRAN_OFFLINE=1
(le serveur public, voir Dockerfile) : jamais de téléchargement, seulement data/.

Chaque recherche tourne dans un processus à part (hitran_tool/worker.py), au plus
MAX_PARALLEL_SEARCHES (défaut 2) à la fois pour tous les visiteurs.
"""
from __future__ import annotations

import dataclasses
import functools
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Scheme

from hitran_tool import SearchConfig, TargetSpecies, worker
from hitran_tool.species import NAME_TO_ID, isotopologue_ids, isotopologue_name

ROOT = Path(__file__).resolve().parent
UNITS = {"ppm": 1e-6, "ppb": 1e-9, "%": 1e-2}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
RED = {"color": "#c22b2b", "fontWeight": 600}


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
def run_search(targets: tuple, config: tuple, with_background: bool):
    """Cached on its (hashable) inputs: changing a display option doesn't recompute."""
    background = str(ROOT / "atm_standard.csv") if with_background else None
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


def targets_from_table(table: pd.DataFrame) -> tuple[list, list]:
    """(targets as (name, mole_fraction, iso_id), errors)."""
    targets, errors = [], []
    for _, row in table.iterrows():
        name = row.get("molécule")
        if not isinstance(name, str) or not name:
            continue
        conc = row.get("concentration")
        if pd.isna(conc) or conc <= 0:
            errors.append(f"{name} : concentration manquante ou nulle")
            continue
        iso = 1 if pd.isna(row.get("isotope")) else int(row.get("isotope"))
        if iso not in isotopologue_ids(name):
            names = ", ".join(f"{k} = {isotopologue_name(name, k)}" for k in isotopologue_ids(name))
            errors.append(f"{name} : isotope {iso} inconnu (possibles : {names})")
            continue
        targets.append((name, float(conc) * UNITS[row.get("unité") or "ppm"], iso))
    if not targets and not errors:
        errors.append("ajoute au moins un gaz à mesurer")
    return targets, errors


def number_input(key, label, value, min_value=None, max_value=None, step=None, format=None, help=None):
    return html.Div([html.Label(label, title=help),
                     dcc.Input(id=key, type="number", value=value, min=min_value, max=max_value, step="any",
                               debounce=True)], className="field")


FIELDS = ["L_km", "delta_sigma", "P_mbar", "T_C", "sigma_min", "sigma_max", "min_abs", "max_abs",
          "interference_max", "with_background"]

app = Dash(__name__, title="Fenêtres spectrales")
server = app.server
# replaces Streamlit's /_stcore/health: used by the Dockerfile HEALTHCHECK and the Kubernetes probes
server.route("/health")(lambda: "ok")

app.layout = html.Div([
    dcc.Checklist([" Mode sombre"], [], id="dark", className="theme-switch"),

    # ------------------------------------------------------------------ inputs ----

    html.H1("Sélection de fenêtres spectrales"),

    html.Div([
        html.Div([
            html.H3("Gaz à mesurer"),
            dash_table.DataTable(
                pd.DataFrame([
                    {"molécule": "CH4", "concentration": 10.0, "unité": "ppm", "isotope": 1},
                    {"molécule": "N2O", "concentration": 1.0, "unité": "ppm", "isotope": 1},
                    {"molécule": "H2O", "concentration": 30.0, "unité": "%", "isotope": 1},
                ]).to_dict("records"),
                editable=True, row_deletable=True, id="gases",
                columns=[
                    {"name": "molécule", "id": "molécule", "presentation": "dropdown"},
                    {"name": "concentration", "id": "concentration", "type": "numeric"},
                    {"name": "unité", "id": "unité", "presentation": "dropdown"},
                    {"name": "isotope", "id": "isotope", "type": "numeric"},
                ],
                dropdown={
                    "molécule": {"options": [{"label": n, "value": n} for n in sorted(NAME_TO_ID)], "clearable": False},
                    "unité": {"options": [{"label": u, "value": u} for u in UNITS], "clearable": False},
                },
                tooltip_header={"isotope": "1 = isotope principal ; ex. CH4 : 2 = ¹³CH4, H2O : 4 = HDO"},
            ),
            html.Button("+", id="add", className="add"),
        ]),

        html.Div([
            html.H3("Instrument"),
            html.Div([
                number_input("L_km", "L effective (km)", value=7.5, min_value=1e-5, format="%g", help="L = c·τ (ringdown)"),
                number_input("delta_sigma", "Balayage (cm⁻¹)", value=0.55, min_value=0.01, step=0.05, format="%g"),
                number_input("P_mbar", "Pression (mbar)", value=30.0, min_value=0.01, format="%g"),
                number_input("T_C", "Température (°C)", value=40.0, format="%g"),
            ], className="pair"),
        ]),

        html.Div([
            html.H3("Recherche"),
            html.Div([
                number_input("sigma_min", "De (cm⁻¹)", value=4394.65, format="%g"),
                number_input("sigma_max", "À (cm⁻¹)", value=4401.65, format="%g",
                             help="toute la gamme 2000–9000 cm⁻¹ : ~2 min"),
                number_input("min_abs", "Absorption min. (%)", value=1.0, min_value=0.01, format="%g",
                             help="limite de détection : SNR requis / SNR du spectre (10 / 1000)"),
                number_input("max_abs", "Absorption max. (%)", value=20.0, min_value=0.1, max_value=99.0, format="%g"),
                number_input("interference_max", "Interférence max. (%)", value=5.0, min_value=0.1, format="%g"),
                dcc.Checklist([" Gaz de fond (42)"], [" Gaz de fond (42)"], id="with_background", className="field"),
            ], className="pair"),
        ]),
    ], className="columns"),

    html.Div(id="errors"),
    html.Button("Lancer la recherche", id="go", className="primary"),
    dcc.Store(id="run"),

    # ------------------------------------------------------------------ results ----

    dcc.Loading([
        html.Div(id="search-error"),
        html.Div(id="results"),
        html.Div([html.Label("Détail de la fenêtre"), dcc.Dropdown(id="choice", clearable=False)], id="choice-box"),
        html.Div(id="detail"),
    ]),
])

# dark mode follows the system setting, like Streamlit did; the switch toggles it in the browser
app.clientside_callback("() => matchMedia('(prefers-color-scheme: dark)').matches ? [' Mode sombre'] : []",
                        Output("dark", "value"), Input("dark", "id"))
app.clientside_callback("v => (document.documentElement.classList.toggle('dark', v.length > 0), 'theme-switch')",
                        Output("dark", "className"), Input("dark", "value"))


@app.callback(Output("gases", "data"), Input("add", "n_clicks"), State("gases", "data"), prevent_initial_call=True)
def add_gas(_, gases):
    return gases + [{"molécule": None, "concentration": None, "unité": "ppm", "isotope": 1}]


def check(gases, L_km, delta_sigma, P_mbar, T_C, sigma_min, sigma_max, min_abs, max_abs, interference_max,
          with_background):
    targets, errors = targets_from_table(pd.DataFrame(gases))
    if None in (L_km, delta_sigma, P_mbar, T_C, sigma_min, sigma_max, min_abs, max_abs, interference_max):
        return targets, errors + ["une valeur est manquante ou invalide"]
    if min_abs >= max_abs:
        errors.append("l'absorption min. doit être plus petite que l'absorption max.")
    if sigma_min >= sigma_max:
        errors.append("la plage spectrale est vide")
    return targets, errors


@app.callback(Output("errors", "children"), Output("go", "disabled"),
              Input("gases", "data"), [Input(f, "value") for f in FIELDS])
def show_errors(*values):
    errors = check(*values)[1]
    return [html.Div(e, className="error") for e in errors], bool(errors)


@app.callback(Output("run", "data"), Output("search-error", "children"),
              Input("go", "n_clicks"), State("gases", "data"), [State(f, "value") for f in FIELDS],
              prevent_initial_call=True)
def search(_, gases, L_km, delta_sigma, P_mbar, T_C, sigma_min, sigma_max, min_abs, max_abs, interference_max,
           with_background):
    targets, errors = check(gases, L_km, delta_sigma, P_mbar, T_C, sigma_min, sigma_max, min_abs, max_abs,
                            interference_max, with_background)
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
        run_search(tuple(targets), tuple(sorted(dataclasses.asdict(config).items())), bool(with_background))
    except ValueError as exc:
        return None, html.Div(f"Paramètres invalides : {exc}", className="error")
    except ServerBusy:
        return None, html.Div("Le serveur est occupé par d'autres recherches : réessaie dans quelques minutes.",
                              className="error")
    except BrokenProcessPool:  # a worker died (e.g. out of memory): start a fresh pool
        search_pool.cache_clear()
        return None, html.Div("La recherche a échoué (serveur surchargé ?) : réessaie, ou réduis la plage spectrale.",
                              className="error")
    return dict(targets=targets, config=sorted(dataclasses.asdict(config).items()),
                with_background=bool(with_background), elapsed=time.time() - t0), None


@app.callback(Output("results", "children"), Output("choice", "options"), Output("choice", "value"),
              Output("choice-box", "hidden"), Output("detail", "children"),
              Input("run", "data"), Input("choice", "value"), Input("dark", "value"))
def results(run, choice, dark):
    if run is None:
        return [], [], None, True, []

    # the browser only keeps the search inputs (dcc.Store); the result comes back from run_search's cache
    result = run_search(tuple(map(tuple, run["targets"])), tuple(map(tuple, run["config"])), run["with_background"])
    labels = [TargetSpecies(n, x, iso).label for n, x, iso in run["targets"]]
    color = {n: COLORS[i % len(COLORS)] for i, n in enumerate(labels)}

    out = [html.Hr()]
    dropped = result.n_rejected_interference
    out.append(dcc.Markdown(f"**{len(result.windows)} fenêtre{'s' * (len(result.windows) > 1)}**"
                            + (f" · {dropped} éliminée{'s' * (dropped > 1)} pour interférence" if dropped else "")
                            + f" · {run['elapsed']:.0f} s"))
    if not result.windows:
        out.append(html.Div("Aucune fenêtre ne passe les critères : essaie une autre plage spectrale ou un seuil "
                            "d'interférence plus élevé.", className="warning"))
        return out, [], None, True, []

    rows, css = [], []
    for i, c in enumerate(result.windows, 1):
        row, style = {"#": i, "centre (cm⁻¹)": round(c.center, 3), "λ (nm)": round(1e7 / c.center, 2)}, {}
        for n in labels:
            row[f"{n} (%)"] = round(100 * (1 - c.transmission[n]), 2)
            if not c.transmission_ok[n]:
                style[f"{n} (%)"] = RED
        row["interférence max. (%)"] = round(100 * max(c.interference.values()), 2)
        rows.append(row)
        css.append(style)
    table = pd.DataFrame(rows)
    out.append(dash_table.DataTable(table.to_dict("records"), style_data_conditional=[
        {"if": {"row_index": i, "column_id": col}, **red} for i, s in enumerate(css) for col, red in s.items()],
        columns=[{"name": col, "id": col, "type": "numeric", "format": Format(precision=3 if col.startswith("centre") else 2,
                  scheme=Scheme.fixed) if col != "#" else Format()} for col in table.columns]))
    out.append(html.P("Absorption de chaque gaz à son pic ; en rouge : hors de la plage visée.", className="caption"))

    # a new search goes back to the first window, as the selectbox did
    choice = choice if ctx.triggered_id != "run" and choice is not None else 0
    options = [{"label": f"#{i + 1} · {result.windows[i].center:.3f} cm⁻¹", "value": i} for i in range(len(result.windows))]
    c = result.windows[choice]
    spectra, others = result.window_spectra[choice], result.interferer_spectra[choice]

    fig = go.Figure()
    fig.add_vrect(x0=c.lo, x1=c.hi, fillcolor="rgba(46,125,79,0.07)", line_width=0)
    for r in [r for r in c.interferers if r.label in others][:3]:
        grid, alpha = others[r.label]
        fig.add_trace(go.Scatter(x=grid, y=alpha, name=r.label, line=dict(color="#999", dash="dash"),
                                 hovertemplate="%{x:.4f} cm⁻¹<br>α = %{y:.2e}<extra>" + r.label + "</extra>"))
    for n, (grid, alpha) in spectra.items():
        fig.add_trace(go.Scatter(x=grid, y=alpha, name=n, line=dict(color=color[n], width=2.5),
                                 hovertemplate="%{x:.4f} cm⁻¹<br>α = %{y:.2e}<extra>" + n + "</extra>"))
        fig.add_trace(go.Scatter(x=[c.peak_position[n]], y=[c.peak_alpha[n]], mode="markers+text", showlegend=False,
                                 marker=dict(symbol="circle-open", size=16, color=color[n], line=dict(width=3)),
                                 text=[f"<b>{n}</b> {100 * (1 - c.transmission[n]):.1f} %"],
                                 textposition="top center", textfont=dict(color=color[n]), hoverinfo="skip"))
    fig.update_yaxes(type="log", title="α (cm⁻¹)", exponentformat="power", dtick=1)  # decades only
    fig.update_xaxes(title="nombre d'onde (cm⁻¹)", tickformat=".2f")
    fig.update_layout(height=460, margin=dict(t=10, b=40), legend=dict(orientation="h", y=-0.2),
                      template="plotly_dark" if dark else "plotly_white", paper_bgcolor="rgba(0,0,0,0)")
    detail = [dcc.Graph(figure=fig)]

    details = pd.DataFrame([{
        "gaz": n, "pic (cm⁻¹)": round(c.peak_position[n], 4),
        "absorption (%)": round(100 * (1 - c.transmission[n]), 2),
        "interférence (%)": round(100 * c.interference.get(n, float("nan")), 2),
        "principal interférent": c.main_interferer.get(n) or "–",
        "T (%/K)": round(c.temperature_coeff.get(n, float("nan")), 2),
    } for n in labels])
    detail.append(dash_table.DataTable(details.to_dict("records"), [{"name": col, "id": col} for col in details.columns]))
    for pair, k in c.ratio_temperature_coeff.items():
        detail.append(html.P(f"Rapport {pair} : {k:+.3f} %/K → {1e3 * c.ratio_temperature_error[pair]:.2g} ‰ pour ±0,1 K",
                             className="caption"))
    return out, options, choice, False, detail


# A search worker (spawn) re-imports this file as __mp_main__ and must not start
# the server, only see the definitions above.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)
