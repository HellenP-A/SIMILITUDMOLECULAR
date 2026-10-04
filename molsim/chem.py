"""Utilidades quimioinformáticas: estandarización, fingerprints, similitud 2D/3D y descriptores.

Las métricas 3D se calculan con el alineamiento gaussiano de forma + color de RDKit
(rdShapeAlign), una implementación abierta del mismo enfoque de OpenEye ROCS:
- shape_3d = Shape Tanimoto del mejor solapamiento
- combo_3d = (Shape Tanimoto + Color Tanimoto) / 2   (ComboScore 0-2 normalizado a 0-1)
Se usa un ensamble de confórmeros (ETKDGv3 + MMFF) para ambas moléculas y se conserva
el par de confórmeros con mayor ComboScore, como hace ROCS.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, QED, rdMolDescriptors, rdShapeAlign
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

FP_RADIUS = 2
FP_BITS = 2048
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
_LARGEST = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)


# ---------------------------------------------------------------- estandarización
def standardize(smiles: str) -> Chem.Mol | None:
    """Parsea, conserva el fragmento orgánico mayor (quita sales/solventes)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _LARGEST.choose(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def canonical(smiles: str) -> str | None:
    mol = standardize(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def random_smiles(mol: Chem.Mol, n: int, seed: int = 0) -> list[str]:
    """SMILES aleatorizados (orden de átomos distinto, misma molécula)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n * 4):
        if len(out) >= n:
            break
        order = rng.permutation(mol.GetNumAtoms()).tolist()
        s = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
        out.append(s)
    return out


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or smiles
    except Exception:
        return smiles


# ---------------------------------------------------------------- 2D
def fingerprint(mol: Chem.Mol):
    return _MORGAN.GetFingerprint(mol)


def tanimoto(fa, fb) -> float:
    return DataStructs.TanimotoSimilarity(fa, fb)


def cosine(fa, fb) -> float:
    return DataStructs.CosineSimilarity(fa, fb)


# ---------------------------------------------------------------- 3D
def conformers(mol: Chem.Mol, n: int = 10, seed: int = 42) -> Chem.Mol | None:
    """Ensamble de confórmeros ETKDGv3 optimizados con MMFF94 (con hidrógenos)."""
    mh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.5
    params.numThreads = 1
    cids = AllChem.EmbedMultipleConfs(mh, numConfs=n, params=params)
    if len(cids) == 0:
        params.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(mh, numConfs=n, params=params)
        if len(cids) == 0:
            return None
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mh, numThreads=1, maxIters=500)
    except Exception:
        pass
    return mh


def shape_inputs(mol3d: Chem.Mol) -> list:
    return [rdShapeAlign.PrepareConformer(mol3d, c.GetId()) for c in mol3d.GetConformers()]


def shape_color(ref_shapes: list, probe_shapes: list) -> tuple[float, float]:
    """Mejor (shape, combo) sobre todos los pares de confórmeros (máximo ComboScore)."""
    best_s, best_c, best_combo = 0.0, 0.0, -1.0
    for rs in ref_shapes:
        for ps in probe_shapes:
            s, c, _ = rdShapeAlign.AlignShapes(rs, ps)
            if s + c > best_combo:
                best_combo, best_s, best_c = s + c, s, c
    return float(best_s), float(best_combo / 2.0)


# ---------------------------------------------------------------- descriptores
def descriptors(mol: Chem.Mol) -> dict:
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    lipinski_violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    return {
        "Peso molecular": round(mw, 2),
        "LogP": round(logp, 2),
        "Donadores H": hbd,
        "Aceptores H": hba,
        "TPSA": round(tpsa, 2),
        "Enlaces rotables": rotb,
        "Anillos aromáticos": arom,
        "Violaciones Lipinski": lipinski_violations,
        "Cumple Veber": bool(rotb <= 10 and tpsa <= 140),
        "QED": round(QED.qed(mol), 3),
    }

