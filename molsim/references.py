"""12 benzimidazoles de referencia ensayados por la Universidad de Salamanca (USAL).

Fuente: tabla "Benzimidazoles ensayados frente a varias spp de Leishmania" (Bz-Alex.docx, USAL).
Corrección: el SMILES de BZ-3 en la tabla omitía el doble enlace C=N del imidazol
(ClC1=CC2=C(NC(NC(=O)...)N2)C=C1 → forma 2,3-dihidro, C13H11ClN4O, 274.71). La figura de la
misma tabla muestra el benzimidazol aromático, por lo que se usa C13H9ClN4O (MW 272.69).
"""

REFERENCES = [
    {"id": "BZ-1", "smiles": "CC1=CC2=C(NC(=N2)C2=CC=C(O2)N(=O)=O)C=C1", "species": "L. donovani", "ic50_um": None, "mw": 243.22},
    {"id": "BZ-2", "smiles": "CC1=CC2=C(NC(NC(=O)C3=NC=CC=C3)=N2)C=C1", "species": "L. donovani", "ic50_um": 1.3, "mw": 252.27},
    {"id": "BZ-3", "smiles": "ClC1=CC2=C(NC(NC(=O)C3=NC=CC=C3)=N2)C=C1", "species": "L. donovani", "ic50_um": 1.9, "mw": 272.69},
    {"id": "BZ-4", "smiles": "CC1=CC=CN=C1C(=O)NC1=NC2=C(N1)C=CC(Cl)=C2", "species": "L. donovani", "ic50_um": None, "mw": 286.72},
    {"id": "BZ-5", "smiles": "ClC1=CC2=C(NC(NC(=O)C3=NC=CC4=C3C=CC=C4)=N2)C=C1", "species": "L. donovani", "ic50_um": None, "mw": 322.75},
    {"id": "BZ-6", "smiles": "CCCCC1=CC=C(C=C1)C(=O)NC1=NC2=C(N1)C=CC(Cl)=C2", "species": "L. donovani", "ic50_um": 0.9, "mw": 327.81},
    {"id": "9d", "smiles": "COC(=O)C1=CC=C(NC(=O)CSC2=NC3=C(C=C(Cl)C=C3)N2CC2=CC(C)=CC(C)=C2)C(Cl)=C1", "species": "L. infantum", "ic50_um": 6.8, "mw": 528.45},
    {"id": "9b", "smiles": "COC(=O)C1=CC=C(NC(=O)CSC2=NC3=C(C=C(Cl)C=C3)N2S(=O)(=O)C2=CC(C)=CC(C)=C2)C=C1", "species": "L. infantum", "ic50_um": 20.3, "mw": 544.04},
    {"id": "141", "smiles": "O=C(NC1=CC(=CC=C1)C1=NC2=C(N1)C=CC=C2)C1=CC=C(C=C1)C1=CC=CC=C1", "species": "L. donovani", "ic50_um": None, "mw": 389.45, "activity": "80 % inhibición a 10 µM"},
    {"id": "7a", "smiles": "CN1C(NCC2=CC(Br)=CC=C2O)=NC2=C1C=CC=C2", "species": "L. mexicana", "ic50_um": 2.62, "mw": 332.20},
    {"id": "7b", "smiles": "NC(=N)NC1=CC2=C(NC(=N2)C2=CC=C(C=C2)C2=CC=CC=C2)C=C1", "species": "L. donovani", "ic50_um": 0.8, "mw": 327.38},
    {"id": "21", "smiles": "O=C(N\\N=C\\C1=CC=C(O1)N(=O)=O)C1=NC2=C(N1)C=CC=C2", "species": "L. amazonensis", "ic50_um": 18.38, "mw": 299.24},
]

# Fármacos de referencia clínica (solo informativos, no se usan como referencias del ranking)
CONTROLS = [
    {"id": "Miltefosina", "smiles": "CCCCCCCCCCCCCCCCOP(=O)([O-])OCC[N+](C)(C)C", "species": "L. mexicana", "ic50_um": 8.6},
    {"id": "Glucantime", "smiles": "CNCC(O)C(O)C(O)C(O)CO.O[Sb](=O)=O", "species": "L. mexicana", "ic50_um": 35.0},
    {"id": "Anfotericina B", "smiles": None, "species": "L. mexicana", "ic50_um": 1.3},
]

METRICS = ["tanimoto_2d", "cosine_2d", "shape_3d", "combo_3d"]
METRIC_LABELS = {
    "tanimoto_2d": "Tanimoto 2D",
    "cosine_2d": "Coseno 2D",
    "shape_3d": "Shape 3D",
    "combo_3d": "Combo 3D",
}
DEFAULT_WEIGHTS = {"tanimoto_2d": 0.35, "cosine_2d": 0.25, "shape_3d": 0.20, "combo_3d": 0.20}
