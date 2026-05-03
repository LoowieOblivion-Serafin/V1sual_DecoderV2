"""
===============================================================================
phase2/train_mindeye.py — Entrenador MindEye fMRI → CLIP ViT-L/14 (BOLD5000)
===============================================================================

Reemplaza al baseline Ridge (`train_adapter.py`) por backbone profundo no lineal
entrenado con InfoNCE + auxiliares (MSE/cos). Contrato: ver `phase2/INTERFACE.md`.

Pipeline:
    1. Carga `Split` vía `loader.load_split` (modo 'bold5000').
    2. Dataset/DataLoader sobre arrays float32. OOM-guard con fallback a bs=32.
    3. Bucle: AdamW + warmup-lineal(500) → CosineAnnealingLR, AMP bf16, clip 1.0.
    4. Selección de mejor checkpoint por *pairwise accuracy* sobre el test set
       (113 repeated, único hold-out disponible en BOLD5000).
    5. Early stopping patience=15. Resume desde checkpoint.
    6. Dump final `embeds_test.pt` en formato {trial_ids, embeddings} idéntico
       al de `train_adapter.py` para compatibilidad con `visual_evaluator.py`.

Uso (Máquina B, RTX 4070 Ti):
    py -3.12 -m phase2.train_mindeye --subject CSI1 --epochs 150 --batch_size 64

Uso (Máquina A dev, RTX 2070 8GB):
    py -3.12 -m phase2.train_mindeye --subject CSI1 --epochs 5 --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset

import config
from .loader import load_split
from .mindeye_models import MindEyeBackbone, MindEyeLoss

logger = logging.getLogger("phase2.train_mindeye")


# ---------------------------------------------------------------------------
# Métricas de validación
# ---------------------------------------------------------------------------

@torch.no_grad()
def pairwise_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    2-AFC pairwise identification accuracy. Para cada i, cuenta como acierto
    si cos(pred_i, target_i) > cos(pred_i, target_j) para todo j != i, promediado
    sobre pares (i, j). Equivalente a la métrica reportada por Koide-Majima 2024.
    """
    pred_n = F.normalize(pred.float(), dim=-1)
    tgt_n = F.normalize(target.float(), dim=-1)
    sim = pred_n @ tgt_n.T                    # (N, N)
    diag = sim.diag().unsqueeze(1)            # (N, 1)
    wins = (diag > sim).float()
    wins.fill_diagonal_(0.0)
    n = sim.shape[0]
    return float(wins.sum().item() / (n * (n - 1)))


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------

def _build_loaders(
    split,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    """Construye loaders train (shuffle) y val (test set, sin shuffle)."""
    train_ds = TensorDataset(
        torch.from_numpy(split.betas_train).float(),
        torch.from_numpy(split.clip_train).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(split.betas_test).float(),
        torch.from_numpy(split.clip_test).float(),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, len(val_ds)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Scheduler: warmup lineal (500 steps) → cosine annealing
# ---------------------------------------------------------------------------

def _build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup_steps, 1),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
        eta_min=1e-6,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


# ---------------------------------------------------------------------------
# Entrenamiento por epoch
# ---------------------------------------------------------------------------

def _train_one_epoch(
    model: MindEyeBackbone,
    loss_fn: MindEyeLoss,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    sums = {"loss": 0.0, "loss_nce": 0.0, "loss_mse": 0.0, "loss_cos": 0.0}
    n_batches = 0

    for voxels, target in loader:
        voxels = voxels.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            pred = model(voxels)
            losses = loss_fn(pred, target)

        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        for k in sums:
            if k in losses:
                sums[k] += float(losses[k].detach().item())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in sums.items()}


@torch.no_grad()
def _evaluate(
    model: MindEyeBackbone,
    loss_fn: MindEyeLoss,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """Devuelve métricas agregadas + (preds, targets) concatenados en CPU."""
    model.eval()
    preds, targets = [], []
    sums = {"loss": 0.0, "loss_nce": 0.0, "loss_mse": 0.0, "loss_cos": 0.0}
    n_batches = 0

    for voxels, target in loader:
        voxels = voxels.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            pred = model(voxels)
            losses = loss_fn(pred, target)
        preds.append(pred.float().cpu())
        targets.append(target.float().cpu())
        for k in sums:
            if k in losses:
                sums[k] += float(losses[k].detach().item())
        n_batches += 1

    preds_cat = torch.cat(preds, dim=0)
    targets_cat = torch.cat(targets, dim=0)
    metrics = {k: v / max(n_batches, 1) for k, v in sums.items()}
    metrics["pairwise_acc"] = pairwise_accuracy(preds_cat, targets_cat)
    return metrics, preds_cat, targets_cat


# ---------------------------------------------------------------------------
# Inferencia ordenada por trial_ids_test (compatibilidad visual_evaluator)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _infer_test_embeddings(
    model: MindEyeBackbone,
    betas_test: np.ndarray,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    X = torch.from_numpy(betas_test).float()
    out = []
    for i in range(0, X.shape[0], batch_size):
        chunk = X[i : i + batch_size].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            y = model(chunk)
        out.append(y.float().cpu())
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Entrena MindEye backbone fMRI→CLIP ViT-L/14")
    ap.add_argument("--subject", choices=list(config.BOLD5000_SUBJECTS), default="CSI1")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=64,
                    help="Default 64 (RTX 4070 Ti). Auto-fallback a 32 si OOM.")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--hidden_dim", type=int, default=4096)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=Path, default=None,
                    help="Ruta a checkpoint .pt para reanudar.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Override del root de salida. Default {phase2_outputs}/mindeye/")
    ap.add_argument("--no-amp", action="store_true",
                    help="Desactiva autocast bf16 (debug en CPU).")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Device ---
    use_cuda = torch.cuda.is_available() and not config.HARDWARE_CONFIG.get("force_cpu", False)
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = (not args.no_amp) and use_cuda
    logger.info(f"Device={device} | AMP-bf16={use_amp}")

    # --- Datos ---
    split = load_split(subject=args.subject, mode="bold5000")
    in_voxels = split.betas_train.shape[1]
    out_dim = int(config.SD_CONFIG["embedding_dim"])
    if split.clip_train.shape[1] != out_dim:
        raise ValueError(
            f"clip_train dim ({split.clip_train.shape[1]}) != SD_CONFIG embedding_dim ({out_dim})"
        )
    logger.info(
        f"[{split.subject}] train={split.betas_train.shape}  "
        f"test={split.betas_test.shape}  out_dim={out_dim}"
    )

    # --- Loaders con OOM-guard ---
    batch_size = args.batch_size
    train_loader, val_loader = _build_loaders(
        split, batch_size, args.num_workers, pin_memory=use_cuda
    )

    # --- Modelo / Loss ---
    model = MindEyeBackbone(
        in_voxels=in_voxels,
        out_dim=out_dim,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
    ).to(device)
    loss_fn = MindEyeLoss().to(device)

    # --- Optimizer / Scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * args.epochs
    scheduler = _build_scheduler(optimizer, args.warmup_steps, total_steps)

    # --- Salida ---
    out_root = args.out_dir or (config.DATA_DIRS["phase2_outputs"] / "mindeye")
    out_dir = out_root / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best_path = out_dir / "best_mindeye_model.pt"
    metrics_path = out_dir / "metrics_mindeye.json"
    embeds_path = out_dir / "embeds_test.pt"

    # --- Estado inicial / Resume ---
    start_epoch = 1
    best_pairwise = -math.inf
    patience_left = args.patience
    history: list[dict] = []

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_pairwise = float(ckpt.get("best_pairwise", -math.inf))
        patience_left = int(ckpt.get("patience_left", args.patience))
        if metrics_path.exists():
            history = json.loads(metrics_path.read_text())
        logger.info(
            f"Resume desde {args.resume}  epoch={start_epoch}  "
            f"best_pairwise={best_pairwise:.4f}  patience_left={patience_left}"
        )

    # --- Loop ---
    for epoch in range(start_epoch, args.epochs + 1):
        try:
            tr = _train_one_epoch(
                model, loss_fn, train_loader, optimizer, scheduler,
                device, use_amp, args.grad_clip,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size <= 32:
                logger.error("OOM con batch_size<=32 — abortando.")
                raise
            new_bs = max(32, batch_size // 2)
            logger.warning(f"OOM en epoch {epoch}. batch_size {batch_size} → {new_bs}. Reintentando.")
            batch_size = new_bs
            train_loader, val_loader = _build_loaders(
                split, batch_size, args.num_workers, pin_memory=use_cuda
            )
            steps_per_epoch = max(len(train_loader), 1)
            total_steps = steps_per_epoch * (args.epochs - epoch + 1)
            scheduler = _build_scheduler(optimizer, args.warmup_steps, total_steps)
            continue

        val, _, _ = _evaluate(model, loss_fn, val_loader, device, use_amp)

        lr_now = optimizer.param_groups[0]["lr"]
        epoch_log = {
            "epoch": epoch,
            "lr": lr_now,
            "batch_size": batch_size,
            "train": tr,
            "val": val,
        }
        history.append(epoch_log)
        metrics_path.write_text(json.dumps(history, indent=2))

        improved = val["pairwise_acc"] > best_pairwise
        if improved:
            best_pairwise = val["pairwise_acc"]
            patience_left = args.patience
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_pairwise": best_pairwise,
                    "patience_left": patience_left,
                    "in_voxels": in_voxels,
                    "out_dim": out_dim,
                    "hidden_dim": args.hidden_dim,
                    "n_blocks": args.n_blocks,
                    "dropout": args.dropout,
                    "subject": args.subject,
                },
                ckpt_best_path,
            )
        else:
            patience_left -= 1

        logger.info(
            f"epoch {epoch:03d}/{args.epochs}  lr={lr_now:.2e}  "
            f"train_loss={tr['loss']:.4f} (nce={tr['loss_nce']:.4f}) | "
            f"val_loss={val['loss']:.4f}  pairwise={val['pairwise_acc']:.4f}  "
            f"{'★ best' if improved else f'patience={patience_left}'}"
        )

        if patience_left <= 0:
            logger.info(f"Early stopping en epoch {epoch} — patience agotado.")
            break

    # --- Inferencia final con el mejor checkpoint ---
    if not ckpt_best_path.exists():
        raise RuntimeError(f"No se generó best checkpoint en {ckpt_best_path}")
    best_blob = torch.load(ckpt_best_path, map_location=device)
    model.load_state_dict(best_blob["model"])
    logger.info(
        f"Cargado best checkpoint epoch={best_blob['epoch']}  "
        f"pairwise={best_blob['best_pairwise']:.4f}"
    )

    embeds_test = _infer_test_embeddings(
        model, split.betas_test, device, use_amp, batch_size=batch_size
    )
    if embeds_test.shape != (len(split.trial_ids_test), out_dim):
        raise ValueError(
            f"Shape de embeds_test inesperado: {tuple(embeds_test.shape)} "
            f"vs ({len(split.trial_ids_test)}, {out_dim})"
        )
    payload = {
        "trial_ids": split.trial_ids_test,
        "embeddings": embeds_test,
    }
    torch.save(payload, embeds_path)

    logger.info(f"Best ckpt    → {ckpt_best_path}")
    logger.info(f"Metrics      → {metrics_path}")
    logger.info(f"Embeds test  → {embeds_path}  shape={tuple(embeds_test.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
