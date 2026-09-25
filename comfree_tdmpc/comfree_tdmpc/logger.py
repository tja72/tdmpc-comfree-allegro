"""CSV logging and plotting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self._keys: list[str] = []

    def log(self, row: dict) -> None:
        self.rows.append(dict(row))
        for k in row:
            if k not in self._keys:
                self._keys.append(k)
        self.flush()

    def flush(self) -> None:
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._keys)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k, "") for k in self._keys})

    def save_json(self, path: str | Path, obj) -> None:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)


def smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k or k <= 1:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")


def new_figure(*args, **kwargs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(*args, **kwargs)
