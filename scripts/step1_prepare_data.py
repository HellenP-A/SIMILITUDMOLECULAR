"""Paso 1 — Preparación de datos y cálculo de las 4 métricas de similitud.

Entradas:
  data/raw/chembl_tanimoto_v2_consolidado.csv  (20,095 compuestos ChEMBL del PIA-02)
  data/raw/chembl35_decoys.csv                 (fármacos aprobados + muestra aleatoria ChEMBL 35)
Salidas:
  data/processed/molecules.csv   (molécula canónica, fuente, scaffold, partición)
  data/processed/pairs.csv.gz    (pares con Tanimoto, Coseno, Shape 3D, Combo 3D)

Mejoras respecto al PIA-02:
  * Partición por scaffold de Bemis-Murcko (validación externa: scaffolds nunca vistos).
  * Decoys diversos de ChEMBL 35 para que el modelo aprenda a discriminar lo no relacionado.
  * 3D real para TODOS los pares (sin valores estimados), con ensamble de confórmeros.
  * Pares de identidad y pares candidato-candidato para un encoder más general.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from molsim import chem  # noqa: E402
from molsim.references import REFERENCES  # noqa: E402
from molsim.tokenizer import tokenize  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")


def split_of(scaf: str, val=0.1, test=0.1) -> str:
    h = int(hashlib.md5(scaf.encode()).hexdigest(), 16) % 10_000 / 10_000
    return "test" if h < test else "val" if h < test + val else "train"


# --------------------------------------------------------------- workers 3D
_REF_SHAPES = None
_N_CONF = 10


def _init(ref_smiles, n_conf):
    global _REF_SHAPES, _N_CONF
    _N_CONF = n_conf
    _REF_SHAPES = []
    for s in ref_smiles:
        m3 = chem.conformers(Chem.MolFromSmiles(s), n_conf)
        _REF_SHAPES.append(chem.shape_inputs(m3))


def _ref_vs_mol(args):
    """Genera confórmeros de una molécula y la compara contra las 12 referencias."""
    idx, smi = args
    m3 = chem.conformers(Chem.MolFromSmiles(smi), _N_CONF)
    if m3 is None:
        return idx, None, None
    shapes = chem.shape_inputs(m3)
    res = [chem.shape_color(rs, shapes) for rs in _REF_SHAPES]
    return idx, res, m3.ToBinary()


def _mol_vs_mol(args):
    k, ba, bb = args
    sa = chem.shape_inputs(Chem.Mol(ba))
    sb = chem.shape_inputs(Chem.Mol(bb))
    return k, chem.shape_color(sa, sb)


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-conf", type=int, default=10)
    ap.add_argument("--random-pairs", type=int, default=20000)
    ap.add_argument("--nn-pairs", type=int, default=20000)
    ap.add_argument("--identity", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="solo para pruebas rápidas")
    ap.add_argument("--max-tokens", type=int, default=126)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(42)

    # ---- 1. moléculas
    leish = pd.read_csv(os.path.join(RAW, "chembl_tanimoto_v2_consolidado.csv"))
    leish = leish.rename(columns={"tanimoto": "pia02_max_tanimoto"})[["chembl_id", "smiles", "pref_name", "pia02_max_tanimoto"]]
    leish["source"] = "chembl_pia02"
    dec = pd.read_csv(os.path.join(RAW, "chembl35_decoys.csv"))[["chembl_id", "smiles", "pref_name", "source"]]
    dec["source"] = "decoy_" + dec["source"]
    refs = pd.DataFrame([{"chembl_id": r["id"], "smiles": r["smiles"], "pref_name": r["id"], "source": "reference"} for r in REFERENCES])
    allm = pd.concat([refs, leish, dec], ignore_index=True)
    if args.limit:
        allm = pd.concat([refs, allm[allm.source != "reference"].sample(args.limit, random_state=0)])

    allm["canonical"] = [chem.canonical(s) for s in allm.smiles]
    allm = allm.dropna(subset=["canonical"])
    allm["n_tokens"] = [len(tokenize(s)) for s in allm.canonical]
    allm = allm[allm.n_tokens <= args.max_tokens]
    allm["heavy_atoms"] = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in allm.canonical]
    allm = allm[allm.heavy_atoms.between(5, 60) | (allm.source == "reference")]
    allm = allm.drop_duplicates("canonical", keep="first").reset_index(drop=True)
    allm["scaffold"] = [chem.scaffold(s) for s in allm.canonical]
    allm["split"] = [("train" if src == "reference" else split_of(sc)) for src, sc in zip(allm.source, allm.scaffold)]
    allm["mol_id"] = np.arange(len(allm))
    print(f"Moléculas: {len(allm)}  ", allm.groupby(["source", "split"]).size().unstack(fill_value=0).to_string())
    allm.drop(columns=["smiles"]).to_csv(os.path.join(OUT, "molecules.csv"), index=False)

    fps = [chem.fingerprint(Chem.MolFromSmiles(s)) for s in allm.canonical]
    ref_ids = allm.index[allm.source == "reference"].tolist()
    cand_ids = allm.index[allm.source != "reference"].tolist()

    # ---- 2. referencias × todas las moléculas (2D + 3D)
    t0 = time.time()
    rows = []
    mol_bins: dict[int, bytes] = {}
    ref_smiles = allm.loc[ref_ids, "canonical"].tolist()
    tasks = [(i, allm.at[i, "canonical"]) for i in ref_ids + cand_ids]
    with Pool(args.workers, initializer=_init, initargs=(ref_smiles, args.n_conf)) as pool:
        for n, (i, res, b) in enumerate(pool.imap_unordered(_ref_vs_mol, tasks, chunksize=16)):
            if res is None:
                continue
            mol_bins[i] = b
            for r_pos, r in enumerate(ref_ids):
                if i in ref_ids and ref_ids.index(i) < r_pos:
                    continue  # ref-ref: solo una vez cada par (incluye identidad)
                kind = "identity" if i == r else ("ref_ref" if i in ref_ids else "ref_mol")
                rows.append((r, i, kind, res[r_pos][0], res[r_pos][1]))
            if n % 2000 == 0:
                print(f"  ref×mol {n}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
    print(f"ref×mol listo en {time.time()-t0:.0f}s")

    # ---- 3. pares candidato-candidato (misma partición): aleatorios + vecinos cercanos
    mm = []
    by_split = {s: [i for i in cand_ids if allm.at[i, "split"] == s and i in mol_bins] for s in ("train", "val", "test")}
    frac = {"train": 0.8, "val": 0.1, "test": 0.1}
    for s, ids in by_split.items():
        n_rand = int(args.random_pairs * frac[s])
        a = rng.choice(ids, n_rand)
        b = rng.choice(ids, n_rand)
        mm += [(x, y, "mol_mol_random") for x, y in zip(a, b) if x != y]
        n_nn = int(args.nn_pairs * frac[s])
        pool_fps = [fps[i] for i in ids]
        pos = {m: k for k, m in enumerate(ids)}
        for x in rng.choice(ids, n_nn, replace=False if n_nn <= len(ids) else True):
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[x], pool_fps))
            sims[pos[x]] = -1
            top = np.argsort(-sims)[:5]
            mm.append((x, ids[int(rng.choice(top))], "mol_mol_nn"))
    t0 = time.time()
    tasks = [(k, mol_bins[a], mol_bins[b]) for k, (a, b, _) in enumerate(mm)]
    res_mm = {}
    with Pool(args.workers) as pool:
        for n, (k, r) in enumerate(pool.imap_unordered(_mol_vs_mol, tasks, chunksize=64)):
            res_mm[k] = r
            if n % 5000 == 0:
                print(f"  mol×mol {n}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
    rows += [(a, b, kind, *res_mm[k]) for k, (a, b, kind) in enumerate(mm)]

    # ---- 4. identidad de candidatos (valor exacto 1.0, sin cálculo)
    for i in rng.choice(cand_ids, min(args.identity, len(cand_ids)), replace=False):
        rows.append((i, i, "identity", 1.0, 1.0))

    pairs = pd.DataFrame(rows, columns=["a", "b", "kind", "shape_3d", "combo_3d"])
    pairs["tanimoto_2d"] = [chem.tanimoto(fps[a], fps[b]) for a, b in zip(pairs.a, pairs.b)]
    pairs["cosine_2d"] = [chem.cosine(fps[a], fps[b]) for a, b in zip(pairs.a, pairs.b)]
    ident = pairs.a == pairs.b
    pairs.loc[ident, ["shape_3d", "combo_3d"]] = 1.0
    pairs["split"] = [allm.at[b, "split"] if allm.at[a, "source"] == "reference" else allm.at[a, "split"] for a, b in zip(pairs.a, pairs.b)]
    pairs = pairs[["a", "b", "kind", "split", "tanimoto_2d", "cosine_2d", "shape_3d", "combo_3d"]].round(5)
    pairs.to_csv(os.path.join(OUT, "pairs.csv.gz"), index=False)
    print(pairs.groupby(["kind", "split"]).size().unstack(fill_value=0).to_string())
    print(pairs.describe().round(3).to_string())


if __name__ == "__main__":
    main()
