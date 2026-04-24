"""
===============================================================================
FASE 2 — INFERENCIA SD 2.1 unCLIP DESDE EMBEDDINGS NSD
===============================================================================

Carga embeddings CLIP ViT-L/14 (768-d) producidos por el adapter fMRI→CLIP
sobre NSD, y los pasa al pipeline SD 2.1 unCLIP para reconstruir imágenes.

PRE-REQUISITO
-------------
El adapter fMRI→CLIP-ViT-L/14 debe estar entrenado y haber producido un
archivo .pt o .npy con la matriz (n_trials, 768) por sujeto. Ese paso vive
en `phase2/train_adapter.py` (pendiente; depende del acceso NSD).

USO
---
    python phase2_run_sd.py --subject sub01 --embeds path/to/embeds.pt
    python phase2_run_sd.py --subject sub01 --embeds path/to/embeds.pt --limit 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from sd_decoder import (
    load_sd_unclip_pipeline,
    reconstruct_from_embedding,
    SD_PRIOR_PROMPTS,
)

GLOBAL_SEED = config.SD_CONFIG["seed"]
INFERENCE_STEPS = config.SD_CONFIG["num_inference_steps"]
GUIDANCE_SCALE = config.SD_CONFIG["guidance_scale"]
EMBED_DIM = config.SD_CONFIG["embedding_dim"]
OUTPUT_ROOT = config.DATA_DIRS["output"]

logger = logging.getLogger("phase2_run_sd")


def load_embeddings(embed_path: Path) -> dict[str, torch.Tensor]:
    """
    Carga embeddings 768-d producidos por el adapter fMRI→CLIP-ViT-L/14.

    Formato esperado:
        - .pt con dict {trial_id: tensor (768,)}, o
        - .pt con tensor (N, 768) + lista paralela de trial_ids almacenada
          como `trial_ids` en el mismo dict.
    """
    if not embed_path.exists():
        raise FileNotFoundError(f"Embeddings no encontrados: {embed_path}")

    data = torch.load(embed_path, map_location="cpu")

    if isinstance(data, dict) and "trial_ids" in data and "embeddings" in data:
        ids = data["trial_ids"]
        emb = data["embeddings"]
        if emb.shape[1] != EMBED_DIM:
            raise ValueError(f"Shape inválido: {emb.shape}, esperado (N, {EMBED_DIM})")
        out = {tid: emb[i] for i, tid in enumerate(ids)}
    elif isinstance(data, dict):
        out = {tid: t.flatten() for tid, t in data.items()}
        bad = [tid for tid, t in out.items() if t.numel() != EMBED_DIM]
        if bad:
            raise ValueError(f"{len(bad)} embeddings con dim != {EMBED_DIM}")
    else:
        raise ValueError(f"Formato no reconocido en {embed_path}")

    logger.info(f"Cargados {len(out)} embeddings desde {embed_path}")
    return out


def run_subject(
    pipeline,
    subject_id: str,
    embeddings: dict[str, torch.Tensor],
    output_dir: Path,
    num_inference_steps: int = INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    limit: int | None = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = list(embeddings.items())
    if limit is not None:
        items = items[:limit]

    saved = 0
    t_start = time.perf_counter()

    for idx, (trial_id, embed) in enumerate(items, 1):
        out_path = output_dir / f"{subject_id}_{trial_id}_sd_unclip.png"
        if out_path.exists():
            logger.info(f"[{subject_id}] ({idx}/{len(items)}) {trial_id} — existe, salto")
            saved += 1
            continue

        try:
            current_prompt = SD_PRIOR_PROMPTS[idx % len(SD_PRIOR_PROMPTS)]
            img = reconstruct_from_embedding(
                pipeline,
                embed,
                prompt=current_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=GLOBAL_SEED,
            )
            img.save(out_path)
            saved += 1
            logger.info(f"[{subject_id}] ({idx}/{len(items)}) {trial_id} → {out_path.name}")
        except Exception as exc:
            logger.error(f"[{subject_id}] Fallo en {trial_id}: {exc}")
            continue

    dt = time.perf_counter() - t_start
    per_img = dt / max(saved, 1)
    logger.info(f"[{subject_id}] {saved}/{len(items)} en {dt:.1f}s ({per_img:.1f}s/img)")
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description="Fase 2 — inferencia SD 2.1 unCLIP desde NSD adapter")
    ap.add_argument("--subject", required=True, help="Identificador NSD (e.g. sub01)")
    ap.add_argument("--embeds", required=True, type=Path,
                    help="Ruta al .pt de embeddings 768-d")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--steps", type=int, default=INFERENCE_STEPS)
    ap.add_argument("--guidance", type=float, default=GUIDANCE_SCALE)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    logger.info(f"Device: {device} | seed={GLOBAL_SEED} | steps={args.steps} | cfg={args.guidance}")

    pipeline = load_sd_unclip_pipeline(device=device, seed=GLOBAL_SEED)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    logger.info(f"Scheduler: {type(pipeline.scheduler).__name__}")

    embeddings = load_embeddings(args.embeds)

    subject_out_dir = OUTPUT_ROOT / args.subject
    total_saved = run_subject(
        pipeline,
        args.subject,
        embeddings,
        subject_out_dir,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        limit=args.limit,
    )

    logger.info(f"Fase 2 completada — {total_saved} imágenes en {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
