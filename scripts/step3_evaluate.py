"""Paso 3 — Evaluación completa del modelo.

Además de R² y Spearman (PIA-02) reporta:
  * MAE, RMSE, Pearson, Kendall por métrica y por tipo de par.
  * Validación EXTERNA por scaffold (moléculas con esqueletos nunca vistos en entrenamiento).
  * Métricas de ranking por referencia: Spearman, NDCG@k y Precision@k del Score final.
  * Línea base: ranking solo con Tanimoto 2D exacto (lo que ofrecen PubChem/ChEMBL).
  * Línea base 3D: regresión lineal de Shape/Combo a partir de las métricas 2D exactas.
  * Prueba de identidad, invariancia a la escritura del SMILES y simetría.
  * Dominio de aplicabilidad: error según la similitud con el set de entrenamiento.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rdkit import Chem, DataStructs  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from molsim import chem  # noqa: E402
from molsim.metrics import ranking, regression  # noqa: E402
from molsim.model import SiameseSimilarity  # noqa: E402
from molsim.references import DEFAULT_WEIGHTS, METRIC_LABELS, METRICS, REFERENCES  # noqa: E402
from molsim.tokenizer import SmilesTokenizer  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")
REPORTS = os.path.join(ROOT, "reports")
FIG = os.path.join(REPORTS, "figures")
W = np.array([DEFAULT_WEIGHTS[m] for m in METRICS])
COLORS = ["#2a6fdb", "#12a37f", "#e07b24", "#b4459b"]

QUALITATIVE = {
    "BZ-6 (referencia)": "CCCCC1=CC=C(C=C1)C(=O)NC1=NC2=C(N1)C=CC(Cl)=C2",
    "Análogo BZ-6 (sin butilo)": "O=C(Nc1nc2cc(Cl)ccc2[nH]1)c1ccccc1",
    "Albendazol (benzimidazol)": "CCCSc1ccc2[nH]c(NC(=O)OC)nc2c1",
    "Mebendazol (benzimidazol)": "COC(=O)Nc1nc2cc(C(=O)c3ccccc3)ccc2[nH]1",
    "2-fenilbencimidazol": "c1ccc(-c2nc3ccccc3[nH]2)cc1",
    "Miltefosina (control antileishmania)": "CCCCCCCCCCCCCCCCOP(=O)([O-])OCC[N+](C)(C)C",
    "Cafeína": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Ibuprofeno": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Aspirina (control negativo)": "CC(=O)Oc1ccccc1C(=O)O",
}


def load_model(device):
    ck = torch.load(os.path.join(MODELS, "best_model.pt"), map_location=device, weights_only=False)
    cfg = dict(ck["config"])
    model = SiameseSimilarity(**cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def run(model, tok, sa: list[str], sb: list[str], device, bs=4096) -> np.ndarray:
    out = []
    for i in range(0, len(sa), bs):
        ta = torch.from_numpy(np.stack([tok.encode(s) for s in sa[i:i + bs]])).to(device)
        tb = torch.from_numpy(np.stack([tok.encode(s) for s in sb[i:i + bs]])).to(device)
        out.append(model(ta, tb).cpu().numpy())
    return np.concatenate(out)


def main():
    os.makedirs(FIG, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, ck = load_model(device)
    tok = SmilesTokenizer.load(os.path.join(MODELS, "tokenizer.json"))
    mols = pd.read_csv(os.path.join(PROC, "molecules.csv"))
    pairs = pd.read_csv(os.path.join(PROC, "pairs.csv.gz"))
    smi = mols.canonical.to_numpy()
    report = {"model": {"parameters": ck["n_params"], "best_epoch": ck["epoch"], "vocab_size": len(tok),
                        "molecules": int(len(mols)), "pairs": int(len(pairs)),
                        "pairs_by_split": pairs.split.value_counts().to_dict(),
                        "molecules_by_split": mols.split.value_counts().to_dict()}}

    # ------------------------------------------------ regresión por partición
    preds = {}
    for split in ("val", "test"):
        d = pairs[pairs.split == split].reset_index(drop=True)
        p = run(model, tok, smi[d.a].tolist(), smi[d.b].tolist(), device)
        preds[split] = (d, p)
        y = d[METRICS].to_numpy()
        r = {m: regression(y[:, k], p[:, k]) for k, m in enumerate(METRICS)}
        r["score"] = regression(y @ W, p @ W)
        r["mean"] = {k: float(np.mean([r[m][k] for m in METRICS])) for k in ("r2", "spearman", "mae", "rmse", "pearson", "kendall")}
        r["by_kind"] = {}
        for kind, g in d.groupby("kind"):
            if kind == "identity" or len(g) < 20:
                continue
            gi = g.index.to_numpy()
            r["by_kind"][kind] = {m: regression(y[gi, k], p[gi, k]) for k, m in enumerate(METRICS)}
        report[split] = r
        print(f"[{split}] R² {r['mean']['r2']:.4f} ρ {r['mean']['spearman']:.4f} MAE {r['mean']['mae']:.4f}")

    # ------------------------------------------------ TTA en test
    d, p = preds["test"]
    rng = np.random.default_rng(0)
    n_aug = 8
    sel = d.index.to_numpy()
    acc = p.copy()
    var_cache = {i: chem.random_smiles(Chem.MolFromSmiles(smi[i]), n_aug - 1, seed=int(i)) for i in np.unique(np.r_[d.a, d.b])}
    for k in range(n_aug - 1):
        sa = [var_cache[i][k % len(var_cache[i])] if var_cache[i] else smi[i] for i in d.a]
        sb = [var_cache[i][k % len(var_cache[i])] if var_cache[i] else smi[i] for i in d.b]
        acc += run(model, tok, sa, sb, device)
    p_tta = acc / n_aug
    y = d[METRICS].to_numpy()
    tta = {m: regression(y[:, k], p_tta[:, k]) for k, m in enumerate(METRICS)}
    tta["mean"] = {k: float(np.mean([tta[m][k] for m in METRICS])) for k in ("r2", "spearman", "mae")}
    report["test_tta"] = tta
    print(f"[test+TTA] R² {tta['mean']['r2']:.4f} ρ {tta['mean']['spearman']:.4f} MAE {tta['mean']['mae']:.4f}")

    # ------------------------------------------------ ranking por referencia (test, ref×mol)
    ref_idx = mols.index[mols.source == "reference"].tolist()
    rm = d[d.kind == "ref_mol"]
    per_ref, base_ref = [], []
    for r in ref_idx:
        g = rm[rm.a == r]
        gi = g.index.to_numpy()
        true_score = y[gi] @ W
        per_ref.append(ranking(true_score, p_tta[gi] @ W))
        base_ref.append(ranking(true_score, g.tanimoto_2d.to_numpy()))
    agg = lambda lst: {k: float(np.mean([x[k] for x in lst])) for k in lst[0]}  # noqa: E731
    report["ranking_test"] = {"model": agg(per_ref), "baseline_tanimoto_only": agg(base_ref),
                              "per_reference": {mols.at[r, "chembl_id"]: x for r, x in zip(ref_idx, per_ref)},
                              "n_candidates_per_reference": int(len(rm) / len(ref_idx))}
    print("[ranking] modelo", {k: round(v, 3) for k, v in report["ranking_test"]["model"].items()})
    print("[ranking] base Tanimoto", {k: round(v, 3) for k, v in report["ranking_test"]["baseline_tanimoto_only"].items()})

    # ------------------------------------------------ línea base 3D desde 2D
    tr = pairs[(pairs.split == "train") & (pairs.kind != "identity")]
    te = d[d.kind != "identity"]
    base3d = {}
    for m in ("shape_3d", "combo_3d"):
        lr = LinearRegression().fit(tr[["tanimoto_2d", "cosine_2d"]], tr[m])
        base3d[m] = {"linear_from_2d": regression(te[m], lr.predict(te[["tanimoto_2d", "cosine_2d"]])),
                     "model": regression(te[m], p_tta[te.index, METRICS.index(m)])}
    report["baseline_3d_from_2d"] = base3d

    # ------------------------------------------------ identidad, invariancia, simetría
    refs_can = [chem.canonical(r["smiles"]) for r in REFERENCES]
    ident_can = run(model, tok, refs_can, refs_can, device)
    rnd = [chem.random_smiles(Chem.MolFromSmiles(s), 2, seed=7) for s in refs_can]
    ident_rnd = run(model, tok, [x[0] for x in rnd], [x[1] for x in rnd], device)
    sym_ab = run(model, tok, smi[d.a[:2000]].tolist(), smi[d.b[:2000]].tolist(), device)
    sym_ba = run(model, tok, smi[d.b[:2000]].tolist(), smi[d.a[:2000]].tolist(), device)
    inv = []
    test_mols = rng.choice(mols.index[mols.split == "test"], 300, replace=False)
    for i in test_mols:
        vs = [smi[i]] + var_cache.get(i, chem.random_smiles(Chem.MolFromSmiles(smi[i]), 7, seed=int(i)))
        pr = run(model, tok, [v for v in vs for _ in refs_can], refs_can * len(vs), device).reshape(len(vs), len(refs_can), 4)
        inv.append(pr.std(axis=0).mean())
    report["sanity"] = {
        "identity_canonical_mean": {m: float(ident_can[:, k].mean()) for k, m in enumerate(METRICS)},
        "identity_random_smiles_mean": {m: float(ident_rnd[:, k].mean()) for k, m in enumerate(METRICS)},
        "identity_score_min": float((ident_rnd @ W).min()),
        "symmetry_max_abs_diff": float(np.abs(sym_ab - sym_ba).max()),
        "smiles_invariance_mean_std": float(np.mean(inv)),
    }
    print("[sanity]", report["sanity"])

    # ------------------------------------------------ dominio de aplicabilidad
    train_fps = [chem.fingerprint(Chem.MolFromSmiles(s)) for s in mols[mols.split == "train"].canonical]
    test_ids = np.unique(rm.b)
    nn = {i: max(DataStructs.BulkTanimotoSimilarity(chem.fingerprint(Chem.MolFromSmiles(smi[i])), train_fps)) for i in test_ids}
    err = np.abs(p_tta[rm.index] - y[rm.index]).mean(axis=1)
    nn_arr = np.array([nn[i] for i in rm.b])
    th = {"alto": 0.5, "medio": 0.3}
    lvl = np.where(nn_arr >= th["alto"], "alto", np.where(nn_arr >= th["medio"], "medio", "bajo"))
    report["applicability_domain"] = {
        "thresholds": th,
        "mae_by_level": {L: float(err[lvl == L].mean()) for L in ("alto", "medio", "bajo") if (lvl == L).any()},
        "pairs_by_level": {L: int((lvl == L).sum()) for L in ("alto", "medio", "bajo")},
    }
    print("[dominio]", report["applicability_domain"])

    # ------------------------------------------------ cualitativo (modelo vs cálculo exacto)
    qual = []
    ref_shapes = [chem.shape_inputs(chem.conformers(Chem.MolFromSmiles(s), 10)) for s in refs_can]
    ref_fps = [chem.fingerprint(Chem.MolFromSmiles(s)) for s in refs_can]
    for name, s in QUALITATIVE.items():
        c = chem.canonical(s)
        mol = Chem.MolFromSmiles(c)
        vs = [c] + chem.random_smiles(mol, 7, seed=0)
        pr = run(model, tok, [v for v in vs for _ in refs_can], refs_can * len(vs), device).reshape(len(vs), len(refs_can), 4).mean(0)
        fp = chem.fingerprint(mol)
        qs = chem.shape_inputs(chem.conformers(mol, 10))
        ex = np.array([[chem.tanimoto(fp, ref_fps[k]), chem.cosine(fp, ref_fps[k]), *chem.shape_color(ref_shapes[k], qs)] for k in range(len(refs_can))])
        sp, se = pr @ W, ex @ W
        best = int(np.argmax(sp))
        qual.append({"compuesto": name, "smiles": c, "mejor_referencia": REFERENCES[best]["id"],
                     "score_modelo": float(sp[best]), "score_exacto": float(se[best]),
                     "mejor_referencia_exacta": REFERENCES[int(np.argmax(se))]["id"],
                     "metricas_modelo": dict(zip(METRICS, map(float, pr[best]))),
                     "metricas_exactas": dict(zip(METRICS, map(float, ex[best])))})
    report["qualitative"] = qual
    for q in qual:
        print(f"  {q['compuesto']:<40} {q['mejor_referencia']:>5}  modelo {q['score_modelo']:.3f}  exacto {q['score_exacto']:.3f}")

    report["pia02_reference"] = {"r2": 0.9353, "spearman": 0.9724, "mae": 0.0180, "pairs": 48462, "parameters": 399300,
                                 "validation": "aleatoria a nivel de par (interna)"}
    with open(os.path.join(REPORTS, "metrics.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)

    # ------------------------------------------------ figuras
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for k, m in enumerate(METRICS):
        ax = axes[k]
        ax.hexbin(y[:, k], p_tta[:, k], gridsize=60, bins="log", cmap="Blues", mincnt=1)
        ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
        ax.set_title(f"{METRIC_LABELS[m]}\nR² {tta[m]['r2']:.3f} · ρ {tta[m]['spearman']:.3f}")
        ax.set_xlabel("Real (RDKit)")
        ax.set_ylabel("Predicho" if k == 0 else "")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Predicho vs real — test externo (scaffolds no vistos)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "scatter_test.png"), dpi=130)

    hist = pd.read_json(os.path.join(MODELS, "training_history.json"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(hist.epoch, hist.train_loss, color=COLORS[0])
    ax[0].set_title("Pérdida de entrenamiento (Huber ponderada)")
    ax[0].set_xlabel("Época")
    ax[0].set_yscale("log")
    ax[1].plot(hist.epoch, hist.val_spearman, color=COLORS[1], label="Spearman")
    ax[1].plot(hist.epoch, hist.val_r2, color=COLORS[2], label="R²")
    ax[1].axvline(ck["epoch"], color="#999", ls="--", lw=1)
    ax[1].set_title("Validación (scaffold)")
    ax[1].set_xlabel("Época")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "training.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    bins = np.linspace(0, 1, 11)
    cen, mae = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (nn_arr >= lo) & (nn_arr < hi)
        if mask.sum() > 30:
            cen.append((lo + hi) / 2)
            mae.append(err[mask].mean())
    ax.bar(cen, mae, width=0.08, color=COLORS[0])
    ax.set_xlabel("Similitud con el vecino más cercano en entrenamiento")
    ax.set_ylabel("MAE promedio")
    ax.set_title("Dominio de aplicabilidad")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "applicability_domain.png"), dpi=130)
    print("Reporte guardado en reports/metrics.json")


if __name__ == "__main__":
    main()
