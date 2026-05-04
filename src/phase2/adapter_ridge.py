"""
Adapter baseline: Ridge regression fMRI → CLIP ViT-L/14 (768-d).

Equivalente al baseline de Takagi-Nishimoto 2023 / Brain-Diffuser. Alpha grande
porque estamos en régimen p > n (V≈10000 vóxeles vs ~9000 trials de train).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class RidgeMetrics:
    r2_macro: float
    cosine_mean: float
    mse: float


class RidgeAdapter:
    def __init__(self, alpha: float = 60_000.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True, copy_X=False)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RidgeAdapter":
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError(f"Shapes inválidos: X={X.shape}, Y={Y.shape}")
        self.model.fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.float32)

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> RidgeMetrics:
        Y_hat = self.predict(X)
        ss_res = ((Y - Y_hat) ** 2).sum(axis=0)
        ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
        r2_per_dim = 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)
        r2_macro = float(r2_per_dim.mean())

        Y_n = Y / np.clip(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12, None)
        Yh_n = Y_hat / np.clip(np.linalg.norm(Y_hat, axis=1, keepdims=True), 1e-12, None)
        cosine_mean = float((Y_n * Yh_n).sum(axis=1).mean())

        mse = float(((Y - Y_hat) ** 2).mean())
        return RidgeMetrics(r2_macro=r2_macro, cosine_mean=cosine_mean, mse=mse)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"alpha": self.alpha, "model": self.model}, path)

    @classmethod
    def load(cls, path: Path) -> "RidgeAdapter":
        blob = joblib.load(path)
        inst = cls(alpha=blob["alpha"])
        inst.model = blob["model"]
        return inst
