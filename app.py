"""App Streamlit — Ranking de similitud molecular multi-métrica contra 12 benzimidazoles anti-Leishmania.

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS

from molsim import chem
from molsim.predictor import Predictor, final_score
from molsim.references import DEFAULT_WEIGHTS, METRIC_LABELS, METRICS, REFERENCES

ROOT = os.path.dirname(__file__)
st.set_page_config(page_title="Similitud Molecular · Benzimidazoles", page_icon="🧬", layout="wide")

EXAMPLES = {
    "BZ-6 (referencia)": "CCCCC1=CC=C(C=C1)C(=O)NC1=NC2=C(N1)C=CC(Cl)=C2",
    "Albendazol": "CCCSc1ccc2[nH]c(NC(=O)OC)nc2c1",
    "Mebendazol": "COC(=O)Nc1nc2cc(C(=O)c3ccccc3)ccc2[nH]1",
    "Análogo de BZ-4": "Cc1cccnc1C(=O)Nc1nc2cc(Br)ccc2[nH]1",
    "2-Fenilbencimidazol": "c1ccc(-c2nc3ccccc3[nH]2)cc1",
    "Miltefosina": "CCCCCCCCCCCCCCCCOP(=O)([O-])OCC[N+](C)(C)C",
    "Aspirina (control negativo)": "CC(=O)Oc1ccccc1C(=O)O",
}
LEVEL_STYLE = {"alto": ("🟢", "Dentro del dominio"), "medio": ("🟡", "Dominio intermedio"), "bajo": ("🔴", "Fuera del dominio")}


@st.cache_resource(show_spinner="Cargando modelo…")
def get_predictor() -> Predictor:
    return Predictor()


def mol_image(mol, size=(360, 260), highlight=None):
    img = Draw.MolToImage(mol, size=size, highlightAtoms=highlight or [])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def mcs_highlight(query, ref, timeout=2):
    res = rdFMCS.FindMCS([query, ref], timeout=timeout, ringMatchesRingOnly=True, completeRingsOnly=True)
    if not res.smartsString:
        return [], [], 0
    patt = Chem.MolFromSmarts(res.smartsString)
    return list(query.GetSubstructMatch(patt)), list(ref.GetSubstructMatch(patt)), res.numAtoms


def fmt_ranking(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["Referencia", *METRICS, "Score", "Incertidumbre", "Especie", "IC50 (µM)"]].rename(columns=METRIC_LABELS)
    return out


def radar(row: pd.Series, name: str, exact: pd.Series | None = None):
    labels = [METRIC_LABELS[m] for m in METRICS]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[row[m] for m in METRICS] + [row[METRICS[0]]], theta=labels + labels[:1],
                                  fill="toself", name="Modelo", line_color="#2a6fdb"))
    if exact is not None and all(m in exact for m in METRICS):
        fig.add_trace(go.Scatterpolar(r=[exact[m] for m in METRICS] + [exact[METRICS[0]]], theta=labels + labels[:1],
                                      name="Cálculo exacto", line=dict(color="#e07b24", dash="dot")))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=330, margin=dict(l=30, r=30, t=40, b=20),
                      title=f"Perfil vs {name}", showlegend=exact is not None)
    return fig


# ------------------------------------------------------------------ barra lateral
pred = get_predictor()
card = pred.card
with st.sidebar:
    st.header("⚙️ Configuración")
    st.caption("Pesos del Score final (se normalizan a 1)")
    weights = {m: st.slider(METRIC_LABELS[m], 0.0, 1.0, DEFAULT_WEIGHTS[m], 0.05) for m in METRICS}
    if sum(weights.values()) == 0:
        weights = DEFAULT_WEIGHTS
    n_aug = st.slider("Aumento en inferencia (escrituras SMILES)", 1, 16, 8,
                      help="Promedia varias escrituras del mismo SMILES; la dispersión se reporta como incertidumbre.")
    verify = st.toggle("Verificar con cálculo exacto (RDKit)", value=False,
                       help="Calcula Tanimoto/Coseno exactos y Shape/Combo 3D con alineamiento de confórmeros (≈1–3 s).")
    st.divider()
    ev = card["evaluation"]["test_tta"]
    st.markdown("**Modelo**")
    st.markdown(f"Siamese CNN · {card['parameters']:,} parámetros · época {card['best_epoch']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("R² test", f"{ev['r2']:.3f}")
    c2.metric("Spearman", f"{ev['spearman']:.3f}")
    c3.metric("MAE", f"{ev['mae']:.3f}")
    st.caption("Test externo: moléculas con scaffolds nunca vistos en entrenamiento.")

st.title("🧬 Similitud molecular multi-métrica")
st.markdown("Ranking de un compuesto contra **12 benzimidazoles anti-*Leishmania*** (USAL) con 4 métricas predichas por una red "
            "neuronal Siamese: **Tanimoto 2D, Coseno 2D, Shape 3D y Combo 3D** (forma + color farmacofórico).")
st.info("Mide **similitud estructural, no actividad biológica**. Es un filtro de priorización; no reemplaza ensayos ni criterio experto.", icon="ℹ️")

tab1, tab2, tab3, tab4 = st.tabs(["🔬 Análisis individual", "📋 Lote", "🧪 Referencias", "📈 Desempeño del modelo"])

# ------------------------------------------------------------------ individual
with tab1:
    col_in, col_ex = st.columns([3, 1])
    example = col_ex.selectbox("Ejemplos", ["—"] + list(EXAMPLES))
    default = EXAMPLES.get(example, EXAMPLES["Albendazol"])
    smiles = col_in.text_input("SMILES del compuesto", value=default)
    if smiles:
        try:
            res = pred.predict(smiles, n_aug=n_aug, weights=weights)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        mol, rk, dom = res["mol"], res["ranking"], res["domain"]
        exact = None
        if verify:
            with st.spinner("Calculando métricas exactas (incluye confórmeros 3D)…"):
                exact = pred.exact(res["canonical"]).set_index("Referencia")

        a, b, c = st.columns([1.2, 1.2, 1.6])
        with a:
            st.image(mol_image(mol), caption=res["canonical"])
        with b:
            icon, label = LEVEL_STYLE[dom["level"]]
            st.markdown(f"#### {icon} {label}")
            st.markdown(f"Similitud con el vecino más cercano del entrenamiento: **{dom['nn_similarity']:.3f}**")
            if dom.get("expected_mae") is not None:
                st.caption(f"Error medio esperado en este rango (test): ±{dom['expected_mae']:.3f}")
            if dom["level"] == "bajo":
                st.warning("Molécula lejana a los datos de entrenamiento: interpretar con cautela. Un score bajo aquí es el comportamiento esperado de un filtro.")
            if res["unk_tokens"]:
                st.warning(f"Tokens no vistos en entrenamiento: {', '.join(res['unk_tokens'])}")
        with c:
            d = chem.descriptors(mol)
            st.dataframe(pd.DataFrame({"Descriptor": list(d), "Valor": [str(v) for v in d.values()]}),
                         hide_index=True, width="stretch", height=280)

        st.subheader("Top 3 referencias")
        cols = st.columns(3)
        for k in range(3):
            row = rk.iloc[k]
            with cols[k]:
                st.metric(f"#{k+1} · {row['Referencia']}", f"{row['Score']:.4f}", help=f"± {row['Incertidumbre']:.4f}")
                st.caption(" · ".join(f"{METRIC_LABELS[m].split()[0]} {row[m]:.4f}" for m in METRICS))

        left, right = st.columns([1, 1])
        top = rk.iloc[0]
        with left:
            st.plotly_chart(radar(top, top["Referencia"], exact.loc[top["Referencia"]] if exact is not None else None), width="stretch")
        with right:
            ref_mol = Chem.MolFromSmiles(top["SMILES referencia"])
            qa, ra, n_mcs = mcs_highlight(mol, ref_mol)
            i1, i2 = st.columns(2)
            i1.image(mol_image(mol, (260, 200), qa), caption="Consulta")
            i2.image(mol_image(ref_mol, (260, 200), ra), caption=top["Referencia"])
            st.caption(f"Subestructura común máxima resaltada ({n_mcs} átomos).")

        st.subheader("Ranking completo")
        table = fmt_ranking(rk)
        st.dataframe(table.style.format({**{METRIC_LABELS[m]: "{:.4f}" for m in METRICS}, "Score": "{:.4f}", "Incertidumbre": "{:.4f}",
                                         "IC50 (µM)": lambda v: "—" if pd.isna(v) else f"{v:g}"})
                     .background_gradient(subset=["Score"], cmap="Blues", vmin=0, vmax=1),
                     width="stretch")

        heat = rk.set_index("Referencia")[METRICS].rename(columns=METRIC_LABELS)
        fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="Blues", zmin=0, zmax=1,
                                   text=np.round(heat.values, 3), texttemplate="%{text}"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), title="Mapa de calor de las 4 métricas", yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")

        if exact is not None:
            st.subheader("Verificación: modelo vs cálculo exacto")
            cmp = rk.set_index("Referencia")[METRICS].join(exact, rsuffix="_exacto")
            exact_score = final_score(exact[METRICS], weights)
            cmp["Score modelo"] = rk.set_index("Referencia")["Score"]
            cmp["Score exacto"] = exact_score
            cmp["|Δ Score|"] = (cmp["Score modelo"] - cmp["Score exacto"]).abs()
            st.dataframe(cmp.style.format("{:.4f}"), width="stretch")
            st.caption(f"Error absoluto medio en las 4 métricas: {np.abs(cmp[METRICS].values - exact[METRICS].loc[cmp.index].values).mean():.4f}")

        csv = table.assign(SMILES_consulta=res["canonical"]).to_csv(index_label="Rank").encode()
        st.download_button("⬇️ Descargar ranking (CSV)", csv, "ranking_similitud.csv", "text/csv")

# ------------------------------------------------------------------ lote
with tab2:
    st.markdown("Pegue varios SMILES (uno por línea, opcionalmente `SMILES nombre`) o suba un CSV con columna `smiles`.")
    txt = st.text_area("SMILES", value="\n".join(f"{s} {n.split(' (')[0].replace(' ', '_')}" for n, s in list(EXAMPLES.items())[1:]), height=180)
    up = st.file_uploader("CSV", type=["csv"])
    if st.button("Calcular ranking del lote", type="primary"):
        if up is not None:
            dfin = pd.read_csv(up)
            col = next(c for c in dfin.columns if c.lower() == "smiles")
            items = [(s, str(dfin.iloc[i].get("name", dfin.iloc[i].get("nombre", f"cmp_{i+1}")))) for i, s in enumerate(dfin[col])]
        else:
            items = []
            for i, line in enumerate(l for l in txt.splitlines() if l.strip()):
                parts = line.split(maxsplit=1)
                items.append((parts[0], parts[1] if len(parts) > 1 else f"cmp_{i+1}"))
        rows, bar = [], st.progress(0.0)
        for k, (s, name) in enumerate(items):
            try:
                r = pred.predict(s, n_aug=min(n_aug, 4), weights=weights)
                best = r["ranking"].iloc[0]
                scores = r["ranking"].set_index("Referencia")["Score"]
                rows.append({"Nombre": name, "SMILES": r["canonical"], "Mejor referencia": best["Referencia"],
                             "Score máximo": best["Score"], "Dominio": r["domain"]["level"],
                             **{f"Score {ref['id']}": scores[ref["id"]] for ref in REFERENCES}})
            except Exception as e:  # noqa: BLE001
                rows.append({"Nombre": name, "SMILES": s, "Mejor referencia": f"Error: {e}"})
            bar.progress((k + 1) / len(items))
        out = pd.DataFrame(rows).sort_values("Score máximo", ascending=False, na_position="last").reset_index(drop=True)
        out.index += 1
        st.dataframe(out.style.format({c: "{:.4f}" for c in out.columns if c.startswith("Score")}, na_rep="—")
                     .background_gradient(subset=["Score máximo"], cmap="Blues", vmin=0, vmax=1), width="stretch")
        st.download_button("⬇️ Descargar lote (CSV)", out.to_csv(index_label="Rank").encode(), "ranking_lote.csv", "text/csv")

# ------------------------------------------------------------------ referencias
with tab3:
    cols = st.columns(4)
    for k, r in enumerate(REFERENCES):
        m = Chem.MolFromSmiles(r["smiles"])
        with cols[k % 4]:
            st.image(mol_image(m, (260, 190)), caption=f"{r['id']} · {r['species']} · " + (r.get("activity") or f"IC50 {r['ic50_um'] if r['ic50_um'] else 'NC'} µM"))
    st.caption("Fuente: tabla de benzimidazoles ensayados (USAL). BZ-3 corregido: el SMILES original omitía el doble enlace C=N "
               "(forma 2,3-dihidro, MW 274.71); se usa el benzimidazol aromático de la figura (MW 272.69). "
               "141: 80 % de inhibición a 10 µM (sin IC50).")
    st.markdown("**Fármacos de referencia clínica** (*L. mexicana*): Miltefosina IC50 8.6 µM · Anfotericina B 1.3 µM · Glucantime 35 µM")

# ------------------------------------------------------------------ desempeño
with tab4:
    ev = card["evaluation"]
    ref = card["pia02_reference"]
    st.markdown("#### Comparación con el PIA-02")
    comp = pd.DataFrame({
        "PIA-02 (documento)": [ref["pairs"], ref["parameters"], ref["r2"], ref["spearman"], ref["mae"], ref["validation"]],
        "Esta versión (test externo + TTA)": [None, card["parameters"], ev["test_tta"]["r2"], ev["test_tta"]["spearman"], ev["test_tta"]["mae"],
                                             "por scaffold (externa)"],
    }, index=["Pares", "Parámetros", "R² promedio", "Spearman promedio", "MAE promedio", "Validación"])
    with open(os.path.join(ROOT, "reports", "metrics.json")) as f:
        rep = json.load(f)
    comp.loc["Pares", "Esta versión (test externo + TTA)"] = rep["model"]["pairs"]
    st.dataframe(comp.astype(str), width="stretch")

    st.markdown("#### Por métrica (test externo)")
    pm = pd.DataFrame(ev["test_per_metric"]).T[["r2", "spearman", "kendall", "pearson", "mae", "rmse"]]
    pm.index = [METRIC_LABELS[m] for m in pm.index]
    st.dataframe(pm.style.format("{:.4f}"), width="stretch")

    st.markdown("#### Calidad del ranking por referencia (Score final)")
    rt = pd.DataFrame(ev["ranking_test"]).T
    rt.index = ["Modelo (4 métricas)", "Solo Tanimoto 2D exacto (tipo PubChem/ChEMBL)"]
    st.dataframe(rt.style.format("{:.4f}"), width="stretch")
    st.caption("NDCG@k y Precision@k: qué tanto el top-k del modelo coincide con el top-k real del Score multi-métrica.")

    s = ev["sanity"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Identidad (score mínimo)", f"{s['identity_score_min']:.4f}")
    c2.metric("Simetría |f(a,b)−f(b,a)|", f"{s['symmetry_max_abs_diff']:.1e}")
    c3.metric("Variación por escritura SMILES", f"{s['smiles_invariance_mean_std']:.4f}")

    for fig in ("scatter_test.png", "training.png", "applicability_domain.png"):
        p = os.path.join(ROOT, "reports", "figures", fig)
        if os.path.exists(p):
            st.image(p)
