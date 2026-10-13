"""Interface graphique pour tester la sélection de fenêtres spectrales.

    uv run streamlit run app.py

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
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hitran_tool import SearchConfig, TargetSpecies, worker
from hitran_tool.species import NAME_TO_ID, isotopologue_ids, isotopologue_name

ROOT = Path(__file__).resolve().parent
UNITS = {"ppm": 1e-6, "ppb": 1e-9, "%": 1e-2}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
RED = "color: #c22b2b; font-weight: 600"


@st.cache_resource
def search_pool() -> ProcessPoolExecutor:
    """Shared by every visitor: at most MAX_PARALLEL_SEARCHES searches run at once, the
    others wait their turn. Separate processes because HAPI's state is global (see
    hitran_tool/worker.py); 'spawn' because forking Streamlit's threads isn't safe."""
    return ProcessPoolExecutor(max_workers=int(os.environ.get("MAX_PARALLEL_SEARCHES", "2")),
                               mp_context=multiprocessing.get_context("spawn"))


@st.cache_data(show_spinner=False, max_entries=8)
def run_search(targets: tuple, config: tuple, with_background: bool):
    """Cached on its (hashable) inputs: changing a display option doesn't recompute."""
    background = str(ROOT / "atm_standard.csv") if with_background else None
    return search_pool().submit(worker.search, targets, config, background).result()


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


def main() -> None:
    st.set_page_config(page_title="Fenêtres spectrales", layout="wide")

    # ------------------------------------------------------------------ inputs ----

    st.title("Sélection de fenêtres spectrales")

    col_gas, col_inst, col_search = st.columns([1.3, 1, 1], gap="large")
    with col_gas:
        st.subheader("Gaz à mesurer")
        gases = st.data_editor(
            pd.DataFrame([
                {"molécule": "CH4", "concentration": 10.0, "unité": "ppm", "isotope": 1},
                {"molécule": "N2O", "concentration": 1.0, "unité": "ppm", "isotope": 1},
                {"molécule": "H2O", "concentration": 30.0, "unité": "%", "isotope": 1},
            ]),
            num_rows="dynamic", hide_index=True, key="gases",
            column_config={
                "molécule": st.column_config.SelectboxColumn(options=sorted(NAME_TO_ID), required=True),
                "concentration": st.column_config.NumberColumn(min_value=0.0, format="%g", required=True),
                "unité": st.column_config.SelectboxColumn(options=list(UNITS), required=True),
                "isotope": st.column_config.NumberColumn(
                    min_value=1, step=1, format="%d",
                    help="1 = isotope principal ; ex. CH4 : 2 = ¹³CH4, H2O : 4 = HDO"),
            },
        )

    with col_inst:
        st.subheader("Instrument")
        a, b = st.columns(2)
        L_km = a.number_input("L effective (km)", value=7.5, min_value=1e-5, format="%g", help="L = c·τ (ringdown)")
        delta_sigma = b.number_input("Balayage (cm⁻¹)", value=0.55, min_value=0.01, step=0.05, format="%g")
        P_mbar = a.number_input("Pression (mbar)", value=30.0, min_value=0.01, format="%g")
        T_C = b.number_input("Température (°C)", value=40.0, format="%g")

    with col_search:
        st.subheader("Recherche")
        a, b = st.columns(2)
        sigma_min = a.number_input("De (cm⁻¹)", value=4394.65, format="%g")
        sigma_max = b.number_input("À (cm⁻¹)", value=4401.65, format="%g",
                                   help="toute la gamme 2000–9000 cm⁻¹ : ~2 min")
        min_abs = a.number_input("Absorption min. (%)", value=1.0, min_value=0.01, format="%g",
                                 help="limite de détection : SNR requis / SNR du spectre (10 / 1000)")
        max_abs = b.number_input("Absorption max. (%)", value=20.0, min_value=0.1, max_value=99.0, format="%g")
        interference_max = a.number_input("Interférence max. (%)", value=5.0, min_value=0.1, format="%g")
        with_background = b.checkbox("Gaz de fond (42)", value=True, help="atm_standard.csv")

    targets, errors = targets_from_table(gases)
    if min_abs >= max_abs:
        errors.append("l'absorption min. doit être plus petite que l'absorption max.")
    if sigma_min >= sigma_max:
        errors.append("la plage spectrale est vide")
    for e in errors:
        st.error(e)

    if st.button("Lancer la recherche", type="primary", width="stretch", disabled=bool(errors), key="go"):
        config = SearchConfig(
            sigma_min=float(sigma_min), sigma_max=float(sigma_max), L_cm=float(L_km) * 1e5,
            delta_sigma=float(delta_sigma), T=float(T_C) + 273.15, P=float(P_mbar) / 1013.25,
            transmission_min=1 - max_abs / 100, transmission_max=1 - min_abs / 100,
            interference_max=float(interference_max) / 100,
        )
        t0 = time.time()
        with st.spinner(f"Recherche sur {sigma_min:g}–{sigma_max:g} cm⁻¹…"):
            try:
                result = run_search(tuple(targets), tuple(sorted(dataclasses.asdict(config).items())),
                                    with_background)
            except ValueError as exc:
                st.error(f"Paramètres invalides : {exc}")
                st.stop()
            except BrokenProcessPool:  # a worker died (e.g. out of memory): start a fresh pool
                search_pool.clear()
                st.error("La recherche a échoué (serveur surchargé ?) : réessaie, ou réduis la plage spectrale.")
                st.stop()
        st.session_state["last_run"] = dict(result=result, targets=targets, config=config, elapsed=time.time() - t0)

    # ------------------------------------------------------------------ results ----

    run = st.session_state.get("last_run")
    if run is None:
        st.stop()

    result, config = run["result"], run["config"]
    labels = [TargetSpecies(n, x, iso).label for n, x, iso in run["targets"]]
    color = {n: COLORS[i % len(COLORS)] for i, n in enumerate(labels)}

    st.divider()
    dropped = result.n_rejected_interference
    st.markdown(f"**{len(result.windows)} fenêtre{'s' * (len(result.windows) > 1)}**"
                + (f" · {dropped} éliminée{'s' * (dropped > 1)} pour interférence" if dropped else "")
                + f" · {run['elapsed']:.0f} s")
    if not result.windows:
        st.warning("Aucune fenêtre ne passe les critères : essaie une autre plage spectrale ou un seuil "
                   "d'interférence plus élevé.")
        st.stop()

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
    st.dataframe(table.style.apply(lambda _: pd.DataFrame([[s.get(col, "") for col in table.columns] for s in css],
                                                          index=table.index, columns=table.columns), axis=None)
                 .format({col: ("{:.3f}" if col.startswith("centre") else "{:.2f}")
                          for col in table.columns if col != "#"}), hide_index=True)
    st.caption("Absorption de chaque gaz à son pic ; en rouge : hors de la plage visée.")

    choice = st.selectbox("Détail de la fenêtre", range(len(result.windows)),
                          format_func=lambda i: f"#{i + 1} · {result.windows[i].center:.3f} cm⁻¹")
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
                      template="plotly_white")
    st.plotly_chart(fig)

    st.dataframe(pd.DataFrame([{
        "gaz": n, "pic (cm⁻¹)": round(c.peak_position[n], 4),
        "absorption (%)": round(100 * (1 - c.transmission[n]), 2),
        "interférence (%)": round(100 * c.interference.get(n, float("nan")), 2),
        "principal interférent": c.main_interferer.get(n) or "–",
        "T (%/K)": round(c.temperature_coeff.get(n, float("nan")), 2),
    } for n in labels]), hide_index=True)
    for pair, k in c.ratio_temperature_coeff.items():
        st.caption(f"Rapport {pair} : {k:+.3f} %/K → {1e3 * c.ratio_temperature_error[pair]:.2g} ‰ pour ±0,1 K")


# Streamlit runs this file as __main__; a search worker (spawn) re-imports it as
# __mp_main__ and must not run the page, only see the definitions above.
if __name__ == "__main__":
    main()
