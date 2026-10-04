"""Tokenizador SMILES basado en expresiones regulares (nivel átomo).

Reconoce átomos de varios caracteres (Br, Cl), átomos entre corchetes ([nH], [N+]),
estereoquímica (@, @@), enlaces, ramas y cierres de anillo (incluido %nn).
"""
from __future__ import annotations

import json
import re

import numpy as np

SMILES_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)
SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
PAD, UNK, BOS, EOS = range(4)


def tokenize(smiles: str) -> list[str]:
    return SMILES_REGEX.findall(smiles)


class SmilesTokenizer:
    def __init__(self, vocab: list[str], max_len: int = 128):
        self.vocab = vocab
        self.max_len = max_len
        self.stoi = {t: i for i, t in enumerate(vocab)}

    @classmethod
    def build(cls, smiles_list, max_len: int = 128, min_freq: int = 1):
        counts: dict[str, int] = {}
        for s in smiles_list:
            for t in tokenize(s):
                counts[t] = counts.get(t, 0) + 1
        tokens = sorted(t for t, c in counts.items() if c >= min_freq)
        return cls(SPECIAL + tokens, max_len)

    def __len__(self):
        return len(self.vocab)

    def encode(self, smiles: str) -> np.ndarray:
        ids = [BOS] + [self.stoi.get(t, UNK) for t in tokenize(smiles)][: self.max_len - 2] + [EOS]
        out = np.zeros(self.max_len, dtype=np.int64)
        out[: len(ids)] = ids
        return out

    def n_tokens(self, smiles: str) -> int:
        return len(tokenize(smiles)) + 2

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "max_len": self.max_len, "regex": SMILES_REGEX.pattern}, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["vocab"], d["max_len"])
