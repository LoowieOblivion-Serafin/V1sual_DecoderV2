"""
===============================================================================
phase2/train_adapter.py — End-to-end Ridge fMRI → CLIP ViT-L/14 (BOLD5000)
===============================================================================

Pipeline:
    1. Carga `Split` vía `loader.load_split` (modo 'bold5000' por defecto).
       Paths se resuelven automáticamente desde `config.BOLD5000_CONFIG`.
    2. Entrena `RidgeAdapter` (baseline Takagi-Nishimoto / Brain-Diffuser).
    3. Evalúa R² / cosine / MSE en train y test (113 repeated set).
    4. Dumpea `ridge_adapter.joblib` + `embeds_test.pt` en el formato que
       consume `phase2_run_sd.py` (keys: 'trial_ids', 'embeddings').

Salida estructurada:
    {phase2_outputs}/adapter/{subject}/
        ├── ridge_adapter.joblib
        └── embeds_test.pt        # {'trial_ids': list[int], 'embeddings': (N,768)}

Uso:
    # Smoke test sintético (sin disco, útil en Máquina A)
    py -3.12 -m phase2.train_adapter --mode mock --snr 0.5

    # Entrenamiento real (Máquina B, requiere .mat + stimuli + clip_targets.pt)
    py -3.12 -m phase2.train_adapter --mode bold5000 --subject CSI1

    # Sobrescribir ruta de CLIP targets
    py -3.12 -m phase2.train_adapter --mode bold5000 --subject CSI1 \\
        --clip-targets /mnt/scratch/bold5000_vitL14.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

import config
from .adapter_ridge import RidgeAdapter
from .loader import load_split

logger = logging.getLogger("phase2.train_adapter")


def _build_loader_kwargs(args: argparse.Namespace) -> dict:
    """
    Traduce los flags del CLI a `loader_kwargs` para `load_split`.

    - mock:     pasa `snr` y `seed` a `make_mock_split`.
    - bold5000: sólo incluye overrides explícitos (clip-targets, rois-mat,
                z-score off). Los no especificados se resuelven dentro del
                facade desde `config.BOLD5000_CONFIG`.
    """
    if args.mode == "mock":
        return {"snr": args.snr, "seed": args.seed}

    kw: dict = {}
    if args.clip_targets is not None:
        kw["clip_targets_pt"] = args.clip_targets
    if args.rois_mat is not None:
        kw["rois_mat"] = args.rois_mat
    if args.no_zscore:
        kw["z_score"] = False
    return kw


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Entrena adapter Ridge fMRI→CLIP-ViT-L/14 (BOLD5000)"
    )
    ap.add_argument(
        "--mode",
        choices=["bold5000", "mock"],
        default="bold5000",
        help="Fuente de datos. 'bold5000' consume BOLD5000_CONFIG; 'mock' genera sintético.",
    )
    ap.add_argument(
        "--subject",
        choices=list(config.BOLD5000_SUBJECTS),
        default="CSI1",
        help="Sujeto BOLD5000 (ignorado en modo mock salvo etiquetado).",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=60_000.0,
        help="Regularización Ridge. p≫n ⇒ alpha grande (default 6e4).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Root de salida. Default: {phase2_outputs}/adapter/",
    )
    # --- overrides modo mock ---
    ap.add_argument("--snr", type=float, default=0.5,
                    help="Solo mock: razón señal/ruido sintética.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Semilla (mock RNG / torch).")
    # --- overrides modo bold5000 ---
    ap.add_argument("--clip-targets", type=Path, default=None,
                    help="Override de clip_targets_pt (BOLD5000_CONFIG).")
    ap.add_argument("--rois-mat", type=Path, default=None,
                    help="Override del .mat de betas ROIs para este subject.")
    ap.add_argument("--no-zscore", action="store_true",
                    help="Desactiva z-score por vóxel (debug).")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    torch.manual_seed(args.seed)

    loader_kwargs = _build_loader_kwargs(args)
    split = load_split(subject=args.subject, mode=args.mode, loader_kwargs=loader_kwargs)
    logger.info(
        f"[{split.subject}/{args.mode}] train={split.betas_train.shape}  "
        f"test={split.betas_test.shape}  Y_dim={split.clip_train.shape[1]}"
    )

    adapter = RidgeAdapter(alpha=args.alpha).fit(split.betas_train, split.clip_train)
    m_tr = adapter.evaluate(split.betas_train, split.clip_train)
    m_te = adapter.evaluate(split.betas_test,  split.clip_test)
    logger.info(
        f"TRAIN  R²={m_tr.r2_macro:+.4f}  cos={m_tr.cosine_mean:+.4f}  mse={m_tr.mse:.4f}"
    )
    logger.info(
    logger.info(
        f"TEST   R²={m_te.r2_macro:+.4f}  cos={m_te.cosine_mean:+.4f}  mse={m_te.mse:.4f}"
    )

    out_root = args.out_dir or (config.DATA_DIRS["phase2_outputs"] / "adapter")
    out_dir = out_root / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    (out_dir / "metrics_test.json").write_text(json.dumps({
        "r2_macro": m_te.r2_macro,
        "cosine_mean": m_te.cosine_mean,
        "mse": m_te.mse,
        "alpha": args.alpha,
        "n_train": split.betas_train.shape[0],
        "n_test": split.betas_test.shape[0],
    }, indent=2))

    adapter_path = out_dir / "ridge_adapter.joblib"
    adapter.save(adapter_path)

    embeds_test = adapter.predict(split.betas_test)
    payload = {
        "trial_ids": split.trial_ids_test,
        "embeddings": torch.from_numpy(embeds_test),
    }
    embeds_path = out_dir / "embeds_test.pt"
    torch.save(payload, embeds_path)

    logger.info(f"Adapter     → {adapter_path}")
    logger.info(f"Embeds test → {embeds_path}  shape={tuple(embeds_test.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
