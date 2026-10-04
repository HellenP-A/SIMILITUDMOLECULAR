"""Paso 4 — Exporta el modelo a ONNX y genera los artefactos de la app (Streamlit y web estática).

Salidas:
  models/similarity.onnx     modelo Siamese completo (tok_a, tok_b) -> 4 métricas
  models/model_card.json     métricas, configuración y umbrales del dominio de aplicabilidad
  models/domain_fps.npy      fingerprints (empaquetados) del set de entrenamiento
  docs/assets/*              copia para la app web de GitHub Pages
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from molsim import chem  # noqa: E402
from molsim.model import SiameseSimilarity  # noqa: E402
from molsim.references import DEFAULT_WEIGHTS, METRIC_LABELS, METRICS, REFERENCES  # noqa: E402
from molsim.tokenizer import SmilesTokenizer  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS = os.path.join(ROOT, "models")
WEB = os.path.join(ROOT, "docs", "assets")
WEB_DOMAIN_SIZE = 6000


def main():
    os.makedirs(WEB, exist_ok=True)
    ck = torch.load(os.path.join(MODELS, "best_model.pt"), map_location="cpu", weights_only=False)
    model = SiameseSimilarity(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    tok = SmilesTokenizer.load(os.path.join(MODELS, "tokenizer.json"))

    # ---- ONNX
    dummy = torch.zeros(2, tok.max_len, dtype=torch.int64)
    onnx_path = os.path.join(MODELS, "similarity.onnx")
    torch.onnx.export(model, (dummy, dummy), onnx_path, input_names=["tok_a", "tok_b"], output_names=["similarity"],
                      dynamic_axes={"tok_a": {0: "n"}, "tok_b": {0: "n"}, "similarity": {0: "n"}}, opset_version=17,
                      dynamo=False)
    # verificación ONNX == PyTorch
    refs = [chem.canonical(r["smiles"]) for r in REFERENCES]
    ta = torch.from_numpy(np.stack([tok.encode(s) for s in refs]))
    tb = torch.from_numpy(np.stack([tok.encode(s) for s in refs[::-1]]))
    with torch.no_grad():
        pt = model(ta, tb).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ox = sess.run(None, {"tok_a": ta.numpy(), "tok_b": tb.numpy()})[0]
    diff = float(np.abs(pt - ox).max())
    print(f"ONNX exportado ({os.path.getsize(onnx_path)/1e6:.2f} MB), diferencia máx. vs PyTorch: {diff:.2e}")
    assert diff < 1e-4

    # ---- dominio de aplicabilidad
    mols = pd.read_csv(os.path.join(ROOT, "data", "processed", "molecules.csv"))
    train = mols[mols.split == "train"].canonical.tolist()
    bits = np.zeros((len(train), 2048), dtype=np.uint8)
    for i, s in enumerate(train):
        bits[i, list(chem.fingerprint(Chem.MolFromSmiles(s)).GetOnBits())] = 1
    packed = np.packbits(bits, axis=1)
    np.save(os.path.join(MODELS, "domain_fps.npy"), packed)
    rng = np.random.default_rng(0)
    web_rows = np.sort(rng.choice(len(packed), min(WEB_DOMAIN_SIZE, len(packed)), replace=False))
    packed[web_rows].tofile(os.path.join(WEB, "domain_fps.bin"))

    # ---- model card
    with open(os.path.join(ROOT, "reports", "metrics.json")) as f:
        rep = json.load(f)
    card = {
        "name": "Siamese CNN multi-métrica — benzimidazoles anti-Leishmania",
        "metrics": METRICS, "metric_labels": METRIC_LABELS, "default_weights": DEFAULT_WEIGHTS,
        "config": ck["config"], "parameters": ck["n_params"], "best_epoch": ck["epoch"], "max_len": tok.max_len,
        "train_molecules": len(train), "web_domain_molecules": int(len(web_rows)),
        "evaluation": {"val": rep["val"]["mean"], "test": rep["test"]["mean"], "test_tta": rep["test_tta"]["mean"],
                       "test_per_metric": {m: rep["test_tta"][m] for m in METRICS},
                       "ranking_test": {"model": rep["ranking_test"]["model"], "baseline_tanimoto_only": rep["ranking_test"]["baseline_tanimoto_only"]},
                       "sanity": rep["sanity"]},
        "applicability_domain": rep["applicability_domain"],
        "pia02_reference": rep["pia02_reference"],
    }
    with open(os.path.join(MODELS, "model_card.json"), "w") as f:
        json.dump(card, f, indent=1, ensure_ascii=False)

    # ---- assets web
    shutil.copy(onnx_path, os.path.join(WEB, "similarity.onnx"))
    shutil.copy(os.path.join(MODELS, "tokenizer.json"), os.path.join(WEB, "tokenizer.json"))
    shutil.copy(os.path.join(MODELS, "model_card.json"), os.path.join(WEB, "model_card.json"))
    web_refs = [dict(r, canonical=chem.canonical(r["smiles"])) for r in REFERENCES]
    with open(os.path.join(WEB, "references.json"), "w") as f:
        json.dump(web_refs, f, indent=1, ensure_ascii=False)
    with open(os.path.join(WEB, "metrics.json"), "w") as f:
        json.dump(rep, f, ensure_ascii=False)
    for fig in ("scatter_test.png", "training.png", "applicability_domain.png"):
        shutil.copy(os.path.join(ROOT, "reports", "figures", fig), os.path.join(WEB, fig))
    print("Artefactos web en docs/assets/")


if __name__ == "__main__":
    main()
