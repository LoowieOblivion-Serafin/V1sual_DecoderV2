"""
===============================================================================
phase2/adapter_ridge_stoch.py — Adapter principal: Ridge Estocástico (BOLD5000)
===============================================================================

Contribución central de la tesis (Cap. 6.7.2, ecs. 6.2-6.3). Toma la salida del
Ridge Lineal y le suma una perturbación Gaussiana isotrópica calibrada, luego
renormaliza sobre la hiperesfera unidad:

        ê_stoch = X · β̂_Ridge + σ · ξ ,     ξ ~ N(0, I_D)        (6.2)
        û_stoch = ê_stoch / ‖ê_stoch‖₂                          (6.3)

`σ` es el único hiperparámetro extra. Se calibra en VALIDACIÓN maximizando
*pairwise accuracy* (2AFC) sobre el espacio CLIP — no minimizando MSE — para
alinear el criterio con la métrica final de discriminación.

Idea (cerrar el Modality Gap a bajo costo): el Ridge Lineal aplana la magnitud
y deja predicciones en zonas de baja densidad del manifold CLIP, donde SD 2.1
unCLIP alucina texto/portadas. Inyectar ruido calibrado + renormalizar empuja
las predicciones hacia zonas más densas del manifold, sin entrenar una red
contrastiva profunda (a diferencia de MindEye). Es la rung intermedia de la
escalera de ablación: Ridge Lineal -> Ridge Estocástico -> MindEye.

Salida (idéntico contrato que train_adapter / train_mindeye):
    {phase2_outputs}/adapter_stoch/{subject}/
        ├── embeds_test.pt     # {'trial_ids': list[int], 'embeddings': (N,768)}
        └── calib_sigma.json   # σ elegido + curva de calibración en validación

Las imágenes las produce luego `visual_evaluator` consumiendo ese embeds_test.pt
(con `--embed-norm none`, porque la renormalización ya la hizo este adapter).

Uso:
    # Smoke sintético (Máquina A, sin disco/GPU)
    py -3.12 -m phase2.adapter_ridge_stoch --mode mock --snr 0.5

    # Real (Máquina B, RTX 4070 Ti)
    py -3.12 -m phase2.adapter_ridge_stoch --mode bold5000 --subject CSI1
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from .adapter_ridge import RidgeAdapter
from .loader import load_split

logger = logging.getLogger("phase2.adapter_ridge_stoch")

# Barrido por defecto de σ (Tabla 6.1 de la tesis).
DEFAULT_SIGMAS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)


# ---------------------------------------------------------------------------
# Núcleo numérico
# ---------------------------------------------------------------------------

def stochastic_transform(
    e_lin: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    renorm: bool = True,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Ecuaciones 6.2-6.3: suma ruido isotrópico y (opcional) renormaliza a la
    hiperesfera unidad. `scale` reescala el resultado final (1.0 = norma unidad,
    tal como la tesis; valores ~12-16 colocan el vector en la escala cruda de
    CLIP ViT-L/14 que SD unCLIP espera — knob de ablación para la Máquina B).
    """
    if e_lin.ndim != 2:
        raise ValueError(f"e_lin debe ser 2D (N, D); recibido {e_lin.shape}")
    noise = rng.standard_normal(e_lin.shape).astype(np.float32) * np.float32(sigma)
    e = e_lin.astype(np.float32) + noise
    if renorm:
        norms = np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-12, None)
        e = e / norms
    return (e * np.float32(scale)).astype(np.float32)


def pairwise_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    2AFC pairwise identification (cosine). Para cada i cuenta como acierto si
    cos(pred_i, target_i) > cos(pred_i, target_j), promediado sobre j != i.
    Idéntico criterio que `train_mindeye.pairwise_accuracy` (versión numpy).
    """
    pn = pred / np.clip(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12, None)
    tn = target / np.clip(np.linalg.norm(target, axis=1, keepdims=True), 1e-12, None)
    sim = pn @ tn.T
    n = sim.shape[0]
    if n < 2:
        return float("nan")
    diag = np.diag(sim)[:, None]
    wins = (diag > sim).astype(np.float64)
    np.fill_diagonal(wins, 0.0)
    return float(wins.sum() / (n * (n - 1)))


@dataclass
class CalibrationResult:
    best_sigma: float
    best_pairwise: float
    table: list[dict]   # [{sigma, pairwise_mean, pairwise_std}, ...]


def calibrate_sigma(
    e_lin_val: np.ndarray,
    Y_val: np.ndarray,
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS,
    n_seeds: int = 5,
    seed: int = 42,
    renorm: bool = True,
) -> CalibrationResult:
    """
    Barre σ y elige el que maximiza pairwise accuracy en validación, promediando
    `n_seeds` realizaciones del ruido (estabiliza la elección frente al azar).
    """
    table: list[dict] = []
    for s in sigmas:
        accs = []
        for k in range(n_seeds):
            rng = np.random.default_rng(seed + k)
            pred = stochastic_transform(e_lin_val, s, rng, renorm=renorm)
            accs.append(pairwise_accuracy(pred, Y_val))
        table.append({
            "sigma": float(s),
            "pairwise_mean": float(np.mean(accs)),
            "pairwise_std": float(np.std(accs)),
        })
    best = max(table, key=lambda r: r["pairwise_mean"])
    return CalibrationResult(
        best_sigma=best["sigma"],
        best_pairwise=best["pairwise_mean"],
        table=table,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class StochasticRidgeAdapter:
    """
    Envuelve un `RidgeAdapter` ya ajustado y aplica la transformación estocástica
    calibrada. Sin estado entrenable propio más allá de σ.
    """

    def __init__(self, ridge: RidgeAdapter, sigma: float, renorm: bool = True, scale: float = 1.0):
        self.ridge = ridge
        self.sigma = float(sigma)
        self.renorm = renorm
        self.scale = float(scale)

    def predict(self, X: np.ndarray, seed: int = 42) -> np.ndarray:
        e_lin = self.ridge.predict(X)
        rng = np.random.default_rng(seed)
        return stochastic_transform(e_lin, self.sigma, rng, renorm=self.renorm, scale=self.scale)


# ---------------------------------------------------------------------------
# Split train/val para calibración (sin tocar el test holdout de 113)
# ---------------------------------------------------------------------------

def _carve_val(betas_train: np.ndarray, clip_train: np.ndarray, val_frac: float, seed: int):
    n = betas_train.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return (betas_train[tr_idx], clip_train[tr_idx],
            betas_train[val_idx], clip_train[val_idx])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Adapter Ridge Estocastico fMRI->CLIP-ViT-L/14 (BOLD5000)"
    )
    ap.add_argument("--mode", choices=["bold5000", "mock"], default="bold5000")
    ap.add_argument("--subject", choices=list(config.BOLD5000_SUBJECTS), default="CSI1")
    ap.add_argument("--alpha", type=float, default=60_000.0, help="Regularizacion Ridge.")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="Fraccion de train reservada para calibrar sigma.")
    ap.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS))
    ap.add_argument("--n-seeds", type=int, default=5, help="Realizaciones de ruido por sigma.")
    ap.add_argument("--no-renorm", action="store_true", help="Desactiva la renorm a hiperesfera.")
    ap.add_argument("--embed-scale", type=float, default=1.0,
                    help="Escala final del embedding. 1.0=unidad (tesis); ~12-16=escala CLIP cruda.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snr", type=float, default=0.5, help="Solo mock.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Root de salida. Default: {phase2_outputs}/adapter_stoch/")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    loader_kwargs = {"snr": args.snr, "seed": args.seed} if args.mode == "mock" else {}
    split = load_split(subject=args.subject, mode=args.mode, loader_kwargs=loader_kwargs)
    renorm = not args.no_renorm
    logger.info(
        f"[{split.subject}/{args.mode}] train={split.betas_train.shape} "
        f"test={split.betas_test.shape} renorm={renorm} scale={args.embed_scale}"
    )

    # 1) Carve val, fit Ridge en train\val, calibra sigma en val
    Xtr, Ytr, Xval, Yval = _carve_val(split.betas_train, split.clip_train, args.val_frac, args.seed)
    ridge_cal = RidgeAdapter(alpha=args.alpha).fit(Xtr, Ytr)
    e_lin_val = ridge_cal.predict(Xval)
    calib = calibrate_sigma(
        e_lin_val, Yval, sigmas=tuple(args.sigmas),
        n_seeds=args.n_seeds, seed=args.seed, renorm=renorm,
    )
    logger.info(f"Calibracion sigma -> best={calib.best_sigma} (pairwise_val={calib.best_pairwise:.4f})")
    for row in calib.table:
        logger.info(f"  sigma={row['sigma']:.3f}  pairwise={row['pairwise_mean']:.4f} +/- {row['pairwise_std']:.4f}")

    # 2) Refit Ridge en TODO el train, aplica sigma elegido al test holdout
    ridge_full = RidgeAdapter(alpha=args.alpha).fit(split.betas_train, split.clip_train)
    adapter = StochasticRidgeAdapter(ridge_full, sigma=calib.best_sigma, renorm=renorm, scale=args.embed_scale)
    embeds_test = adapter.predict(split.betas_test, seed=args.seed)

    if embeds_test.shape != (len(split.trial_ids_test), split.clip_test.shape[1]):
        raise ValueError(f"Shape embeds_test inesperado: {embeds_test.shape}")

    # 3) Persistencia (torch sólo para el .pt, igual que train_adapter)
    import torch
    out_root = args.out_dir or (config.DATA_DIRS["phase2_outputs"] / "adapter_stoch")
    out_dir = out_root / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"trial_ids": split.trial_ids_test, "embeddings": torch.from_numpy(embeds_test)},
        out_dir / "embeds_test.pt",
    )
    (out_dir / "calib_sigma.json").write_text(json.dumps({
        "best_sigma": calib.best_sigma,
        "best_pairwise_val": calib.best_pairwise,
        "renorm": renorm,
        "embed_scale": args.embed_scale,
        "alpha": args.alpha,
        "table": calib.table,
    }, indent=2))

    logger.info(f"Embeds test -> {out_dir / 'embeds_test.pt'}  shape={tuple(embeds_test.shape)}")
    logger.info(f"Calib       -> {out_dir / 'calib_sigma.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
