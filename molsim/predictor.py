"""Inferencia con el modelo exportado a ONNX (no requiere PyTorch ni OpenEye).

Incluye:
  * Aumento en inferencia (TTA): promedia N escrituras SMILES de la consulta y reporta
    la desviación estándar como medida de incertidumbre.
  * Dominio de aplicabilidad: similitud Tanimoto con el vecino más cercano del set de
    entrenamiento; indica cuándo la predicción es extrapolación.
  * Cálculo exacto opcional (RDKit) de las 4 métricas para verificar al modelo.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

from . import chem
from .references import DEFAULT_WEIGHTS, METRICS, REFERENCES
from .tokenizer import SmilesTokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def final_score(df: pd.DataFrame, weights: dict = DEFAULT_WEIGHTS) -> pd.Series:
    tot = sum(weights.values()) or 1.0
    return sum(df[m] * w for m, w in weights.items()) / tot


class Predictor:
    def __init__(self, model_dir: str = MODEL_DIR):
        import onnxruntime as ort

        self.session = ort.InferenceSession(os.path.join(model_dir, "similarity.onnx"), providers=["CPUExecutionProvider"])
        self.tok = SmilesTokenizer.load(os.path.join(model_dir, "tokenizer.json"))
        with open(os.path.join(model_dir, "model_card.json")) as f:
            self.card = json.load(f)
        self.refs = [dict(r, canonical=chem.canonical(r["smiles"])) for r in REFERENCES]
        self.ref_tokens = np.stack([self.tok.encode(r["canonical"]) for r in self.refs])
        self.ref_fps = [chem.fingerprint(Chem.MolFromSmiles(r["canonical"])) for r in self.refs]
        packed = np.load(os.path.join(model_dir, "domain_fps.npy"))
        self.domain_fps = [self._from_bits(row) for row in np.unpackbits(packed, axis=1)]
        self._ref_shapes = None

    @staticmethod
    def _from_bits(bits: np.ndarray):
        fp = DataStructs.ExplicitBitVect(len(bits))
        fp.SetBitsFromList(np.flatnonzero(bits).tolist())
        return fp

    # ------------------------------------------------------------ modelo
    def _run(self, tok_a: np.ndarray, tok_b: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"tok_a": tok_a.astype(np.int64), "tok_b": tok_b.astype(np.int64)})[0]

    def predict(self, smiles: str, n_aug: int = 8, weights: dict = DEFAULT_WEIGHTS) -> dict:
        mol = chem.standardize(smiles)
        if mol is None:
            raise ValueError(f"SMILES inválido: {smiles}")
        can = Chem.MolToSmiles(mol)
        variants = [can] + chem.random_smiles(mol, max(n_aug - 1, 0), seed=0)
        q = np.stack([self.tok.encode(v) for v in variants])  # (V,L)
        n_ref = len(self.refs)
        tok_a = np.repeat(q, n_ref, axis=0)
        tok_b = np.tile(self.ref_tokens, (len(variants), 1))
        pred = self._run(tok_a, tok_b).reshape(len(variants), n_ref, len(METRICS))
        mean, std = pred.mean(axis=0), pred.std(axis=0)

        df = pd.DataFrame(mean, columns=METRICS)
        df.insert(0, "Referencia", [r["id"] for r in self.refs])
        df["Score"] = final_score(df, weights)
        df["Incertidumbre"] = std.mean(axis=1)
        df["Especie"] = [r["species"] for r in self.refs]
        df["IC50 (µM)"] = [r["ic50_um"] for r in self.refs]
        df["SMILES referencia"] = [r["canonical"] for r in self.refs]
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df.index += 1
        return {"canonical": can, "mol": mol, "ranking": df, "domain": self.domain(mol), "unk_tokens": self._unk(can)}

    def _unk(self, smiles: str) -> list[str]:
        from .tokenizer import tokenize

        return sorted({t for t in tokenize(smiles) if t not in self.tok.stoi})

    # ------------------------------------------------------------ dominio de aplicabilidad
    def domain(self, mol: Chem.Mol) -> dict:
        fp = chem.fingerprint(mol)
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, self.domain_fps))
        nn = float(sims.max())
        th = self.card["applicability_domain"]["thresholds"]
        level = "alto" if nn >= th["alto"] else "medio" if nn >= th["medio"] else "bajo"
        return {"nn_similarity": nn, "level": level, "expected_mae": self.card["applicability_domain"]["mae_by_level"].get(level)}

    # ------------------------------------------------------------ cálculo exacto (verificación)
    def exact(self, smiles: str, with_3d: bool = True, n_conf: int = 10) -> pd.DataFrame:
        mol = chem.standardize(smiles)
        fp = chem.fingerprint(mol)
        rows = []
        if with_3d:
            if self._ref_shapes is None:
                self._ref_shapes = [chem.shape_inputs(chem.conformers(Chem.MolFromSmiles(r["canonical"]), n_conf)) for r in self.refs]
            m3 = chem.conformers(mol, n_conf)
            q_shapes = chem.shape_inputs(m3) if m3 is not None else None
        for k, r in enumerate(self.refs):
            row = {"Referencia": r["id"], "tanimoto_2d": chem.tanimoto(fp, self.ref_fps[k]), "cosine_2d": chem.cosine(fp, self.ref_fps[k])}
            if with_3d and q_shapes is not None:
                row["shape_3d"], row["combo_3d"] = chem.shape_color(self._ref_shapes[k], q_shapes)
            rows.append(row)
        return pd.DataFrame(rows)
