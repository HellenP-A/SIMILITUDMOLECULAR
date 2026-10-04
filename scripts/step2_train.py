"""Paso 2 — Tokenización + entrenamiento de la red Siamese CNN.

Mejoras respecto al PIA-02:
  * Aumento de datos con SMILES aleatorizados (la misma molécula escrita de otra forma),
    lo que enseña al modelo que la representación no cambia la molécula.
  * Pares de identidad escritos con SMILES distintos en cada lado (identidad ≈ 1.0 real).
  * Muestreo balanceado por rango de similitud (los pares muy similares son escasos pero
    son los que más importan para el ranking).
  * Early stopping por Spearman promedio de validación (la métrica que importa para rankear).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from molsim import chem  # noqa: E402
from molsim.metrics import regression  # noqa: E402
from molsim.model import SiameseSimilarity, count_parameters  # noqa: E402
from molsim.references import METRICS  # noqa: E402
from molsim.tokenizer import SmilesTokenizer  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")
LOSS_W = torch.tensor([1.5, 1.2, 0.8, 1.0])


def build_variants(smiles: list[str], tok: SmilesTokenizer, k: int) -> np.ndarray:
    """Para cada molécula: variante 0 = SMILES canónico, 1..k-1 = SMILES aleatorizados."""
    arr = np.zeros((len(smiles), k, tok.max_len), dtype=np.int16)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        variants = [s] + chem.random_smiles(mol, k - 1, seed=i)
        variants += [s] * (k - len(variants))
        for j, v in enumerate(variants):
            arr[i, j] = tok.encode(v)
    return arr


@torch.no_grad()
def predict(model, var, a, b, device, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(a), bs):
        ta = torch.from_numpy(var[a[i:i + bs], 0].astype(np.int64)).to(device)
        tb = torch.from_numpy(var[b[i:i + bs], 0].astype(np.int64)).to(device)
        out.append(model(ta, tb).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--samples-per-epoch", type=int, default=250_000)
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--p-aug", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(MODELS, exist_ok=True)

    mols = pd.read_csv(os.path.join(PROC, "molecules.csv"))
    pairs = pd.read_csv(os.path.join(PROC, "pairs.csv.gz"))
    smiles = mols.canonical.tolist()

    # ---- tokenizador (vocabulario solo de train + referencias, incluyendo variantes aleatorias)
    train_mols = mols[mols.split == "train"].canonical.tolist()
    vocab_src = train_mols + [r for i, s in enumerate(train_mols[:5000]) for r in chem.random_smiles(Chem.MolFromSmiles(s), 2, seed=i)]
    tok = SmilesTokenizer.build(vocab_src, max_len=128)
    tok.save(os.path.join(MODELS, "tokenizer.json"))
    print(f"Vocabulario: {len(tok)} tokens | dispositivo: {device}")

    t0 = time.time()
    var = build_variants(smiles, tok, args.variants)
    print(f"Variantes SMILES: {var.shape} en {time.time()-t0:.0f}s")

    tr = pairs[pairs.split == "train"].reset_index(drop=True)
    va = pairs[pairs.split == "val"].reset_index(drop=True)
    y_tr = tr[METRICS].to_numpy(np.float32)
    a_tr, b_tr = tr.a.to_numpy(), tr.b.to_numpy()

    # ---- pesos de muestreo: inverso de la raíz de la frecuencia por bin de Tanimoto
    bins = np.minimum((tr.tanimoto_2d.to_numpy() / 0.05).astype(int), 19)
    freq = np.bincount(bins, minlength=20).astype(float)
    w = 1.0 / np.sqrt(freq[bins])
    w /= w.sum()

    model = SiameseSimilarity(len(tok)).to(device)
    n_params = count_parameters(model)
    print(f"Parámetros entrenables: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = args.samples_per_epoch // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch, pct_start=0.1)
    loss_w = LOSS_W.to(device)

    history, best, bad = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        idx = rng.choice(len(tr), steps_per_epoch * args.batch, p=w)
        run = 0.0
        for s in range(steps_per_epoch):
            bi = idx[s * args.batch:(s + 1) * args.batch]
            va_ = np.where(rng.random(len(bi)) < args.p_aug, rng.integers(1, args.variants, len(bi)), 0)
            vb_ = np.where(rng.random(len(bi)) < args.p_aug, rng.integers(1, args.variants, len(bi)), 0)
            ident = a_tr[bi] == b_tr[bi]
            vb_[ident] = rng.integers(0, args.variants, ident.sum())  # identidad: dos escrituras distintas
            ta = torch.from_numpy(var[a_tr[bi], va_].astype(np.int64)).to(device)
            tb = torch.from_numpy(var[b_tr[bi], vb_].astype(np.int64)).to(device)
            y = torch.from_numpy(y_tr[bi]).to(device)
            pred = model(ta, tb)
            loss = (F.smooth_l1_loss(pred, y, reduction="none", beta=0.05) * loss_w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()

        p = predict(model, var, va.a.to_numpy(), va.b.to_numpy(), device)
        yv = va[METRICS].to_numpy()
        per = {m: regression(yv[:, k], p[:, k]) for k, m in enumerate(METRICS)}
        mean_sp = float(np.mean([per[m]["spearman"] for m in METRICS]))
        mean_r2 = float(np.mean([per[m]["r2"] for m in METRICS]))
        mean_mae = float(np.mean([per[m]["mae"] for m in METRICS]))
        history.append({"epoch": epoch, "train_loss": run / steps_per_epoch, "val_r2": mean_r2, "val_spearman": mean_sp,
                        "val_mae": mean_mae, **{f"val_r2_{m}": per[m]["r2"] for m in METRICS},
                        "lr": sched.get_last_lr()[0], "seconds": time.time() - t0})
        flag = ""
        if mean_sp > best:
            best, bad, flag = mean_sp, 0, " *"
            torch.save({"state_dict": model.state_dict(), "config": model.config, "metrics": METRICS,
                        "epoch": epoch, "val": per, "n_params": n_params, "vocab": tok.vocab, "max_len": tok.max_len},
                       os.path.join(MODELS, "best_model.pt"))
        else:
            bad += 1
        print(f"época {epoch:3d} | loss {run/steps_per_epoch:.5f} | val R² {mean_r2:.4f} ρ {mean_sp:.4f} MAE {mean_mae:.4f} "
              f"| {time.time()-t0:.0f}s{flag}", flush=True)
        with open(os.path.join(MODELS, "training_history.json"), "w") as f:
            json.dump(history, f, indent=1)
        if bad >= args.patience:
            print("Early stopping.")
            break
    print(f"Mejor Spearman de validación: {best:.4f}")


if __name__ == "__main__":
    main()
