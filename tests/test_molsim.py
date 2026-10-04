import os

import numpy as np
import pytest
from rdkit import Chem

from molsim import chem
from molsim.references import REFERENCES
from molsim.tokenizer import SmilesTokenizer, tokenize

ROOT = os.path.join(os.path.dirname(__file__), "..")
HAS_MODEL = os.path.exists(os.path.join(ROOT, "models", "similarity.onnx"))


def test_tokenizer_keeps_multichar_atoms():
    assert tokenize("ClC[nH]Br%12") == ["Cl", "C", "[nH]", "Br", "%12"]


def test_tokenizer_roundtrip_all_references():
    tok = SmilesTokenizer.build([r["smiles"] for r in REFERENCES])
    for r in REFERENCES:
        ids = tok.encode(r["smiles"])
        toks = [tok.vocab[i] for i in ids if i > 3]
        assert "".join(toks) == r["smiles"]


def test_reference_weights_match_document():
    for r in REFERENCES:
        mw = chem.descriptors(chem.standardize(r["smiles"]))["Peso molecular"]
        assert abs(mw - r["mw"]) < 0.05, r["id"]


def test_random_smiles_same_molecule():
    mol = Chem.MolFromSmiles(REFERENCES[5]["smiles"])
    can = Chem.MolToSmiles(mol)
    for s in chem.random_smiles(mol, 5):
        assert Chem.MolToSmiles(Chem.MolFromSmiles(s)) == can


def test_standardize_strips_salts():
    assert chem.canonical("CC(=O)Oc1ccccc1C(=O)O.[Na+].[Cl-]") == chem.canonical("CC(=O)Oc1ccccc1C(=O)O")


def test_3d_identity_and_symmetry():
    a = chem.conformers(chem.standardize(REFERENCES[3]["smiles"]), 5)
    b = chem.conformers(chem.standardize(REFERENCES[5]["smiles"]), 5)
    sa, sb = chem.shape_inputs(a), chem.shape_inputs(b)
    assert chem.shape_color(sa, sa)[0] == pytest.approx(1.0, abs=1e-3)
    ab, ba = chem.shape_color(sa, sb), chem.shape_color(sb, sa)
    assert ab[0] == pytest.approx(ba[0], abs=0.05)


@pytest.mark.skipif(not HAS_MODEL, reason="modelo no exportado")
def test_predictor_identity_and_discrimination():
    from molsim.predictor import Predictor

    p = Predictor()
    bz6 = p.predict(REFERENCES[5]["smiles"])["ranking"]
    assert bz6.iloc[0]["Referencia"] == "BZ-6"
    assert bz6.iloc[0]["Score"] > 0.95
    aspirin = p.predict("CC(=O)Oc1ccccc1C(=O)O")["ranking"]
    assert aspirin["Score"].max() < 0.5
    assert np.isfinite(aspirin["Incertidumbre"]).all()
