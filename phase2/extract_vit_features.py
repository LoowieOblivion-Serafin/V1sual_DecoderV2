"""
Extracción de embeddings CLIP ViT-L/14 (768-d) sobre stimuli BOLD5000.

Produce la **Variable Y** (target) de la regresión Ridge fMRI→CLIP. Se ejecuta
en paralelo a la descarga de los betas BOLD5000 desde OpenNeuro.

Uso:
    python -m phase2.extract_vit_features \\
        --stimuli-dir data_bold5000/stimuli \\
        --out phase2_outputs/clip_targets/bold5000_vitL14.pt

    # Smoke test rápido:
    python -m phase2.extract_vit_features \\
        --stimuli-dir data_bold5000/stimuli --limit 32

Output (.pt):
    {
        "model_id":   "openai/clip-vit-large-patch14",
        "dim":        768,
        "filenames":  list[str],     # rutas relativas a stimuli-dir
        "embeddings": tensor (N, 768) float32 (NO L2-normalizado),
    }

Nota: stimuli BOLD5000 NO vienen con `ds001499` (licencias ImageNet/COCO/SUN).
Descargar zip aparte de https://bold5000-dataset.github.io/website/download.html
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPProcessor

logger = logging.getLogger("phase2.extract_vit_features")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_MODEL = "openai/clip-vit-large-patch14"
HF_CACHE_DIR = Path("models_hf")


def find_images(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    files.sort()
    return files


def pick_dtype(device: torch.device, force_fp32: bool) -> torch.dtype:
    if force_fp32 or device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_clip(model_id: str, device: torch.device, dtype: torch.dtype):
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading CLIP {model_id}  device={device}  dtype={dtype}")
    model = CLIPVisionModelWithProjection.from_pretrained(model_id, cache_dir=str(HF_CACHE_DIR))
    model = model.to(device=device, dtype=dtype).eval()
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=str(HF_CACHE_DIR))
    return model, processor


def encode_batch(model, processor, paths: list[Path], device, dtype) -> torch.Tensor:
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    with torch.no_grad():
        out = model(pixel_values=pixel_values)
        feats = out.image_embeds
    return feats.float().cpu()


def extract(
    stimuli_dir: Path,
    out_path: Path,
    model_id: str = DEFAULT_MODEL,
    batch_size: int = 32,
    limit: int | None = None,
    device_str: str | None = None,
    force_fp32: bool = False,
) -> dict:
    if not stimuli_dir.exists():
        raise FileNotFoundError(f"Stimuli dir no existe: {stimuli_dir}")

    all_files = find_images(stimuli_dir)
    if not all_files:
        raise RuntimeError(f"No se hallaron imágenes en {stimuli_dir} (ext: {IMAGE_EXTS})")

    rel_files = [str(p.relative_to(stimuli_dir)).replace("\\", "/") for p in all_files]

    already_done: dict[str, torch.Tensor] = {}
    if out_path.exists():
        try:
            blob = torch.load(out_path, map_location="cpu")
            for fname, idx in zip(blob["filenames"], range(len(blob["filenames"]))):
                already_done[fname] = blob["embeddings"][idx]
            logger.info(f"Resume: {len(already_done)} embeddings cargados de {out_path}")
        except Exception as exc:
            logger.warning(f"No se pudo reanudar desde {out_path}: {exc}. Recomputo todo.")
            already_done = {}

    pending_idx = [i for i, fn in enumerate(rel_files) if fn not in already_done]
    if limit is not None:
        pending_idx = pending_idx[:limit]
    if not pending_idx:
        logger.info("Nada pendiente. Saliendo.")
        return {"filenames": rel_files, "embeddings": torch.stack([already_done[f] for f in rel_files])}

    logger.info(f"Total imgs={len(rel_files)}  ya hechas={len(already_done)}  pendientes={len(pending_idx)}")

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype(device, force_fp32=force_fp32)
    model, processor = load_clip(model_id, device, dtype)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n_done = 0
    save_every = max(batch_size * 10, 256)

    for batch_start in range(0, len(pending_idx), batch_size):
        chunk = pending_idx[batch_start:batch_start + batch_size]
        paths = [all_files[i] for i in chunk]
        try:
            feats = encode_batch(model, processor, paths, device, dtype)
        except Exception as exc:
            logger.error(f"Batch falló en {paths[0].name}…{paths[-1].name}: {exc}")
            continue
        for fname, vec in zip([rel_files[i] for i in chunk], feats):
            already_done[fname] = vec
        n_done += len(chunk)

        if n_done % save_every < batch_size or batch_start + batch_size >= len(pending_idx):
            _save(out_path, model_id, rel_files, already_done)
            elapsed = time.perf_counter() - t0
            ips = n_done / max(elapsed, 1e-9)
            eta = (len(pending_idx) - n_done) / max(ips, 1e-9)
            logger.info(f"{n_done}/{len(pending_idx)}  {ips:.1f} img/s  ETA {eta/60:.1f}min")

    _save(out_path, model_id, rel_files, already_done)
    logger.info(f"Listo. {len(already_done)} embeddings → {out_path}")
    return {"filenames": rel_files, "embeddings": torch.stack([already_done[f] for f in rel_files if f in already_done])}


def _save(out_path: Path, model_id: str, rel_files: list[str], embeds: dict[str, torch.Tensor]) -> None:
    ordered_files = [f for f in rel_files if f in embeds]
    matrix = torch.stack([embeds[f] for f in ordered_files])
    payload = {
        "model_id": model_id,
        "dim": int(matrix.shape[1]),
        "filenames": ordered_files,
        "embeddings": matrix,
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extrae CLIP ViT-L/14 (768-d) sobre stimuli BOLD5000")
    ap.add_argument("--stimuli-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("phase2_outputs/clip_targets/bold5000_vitL14.pt"))
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="Solo primeras N pendientes (debug)")
    ap.add_argument("--device", default=None, help="cuda | cpu | cuda:0 …")
    ap.add_argument("--fp32", action="store_true", help="Fuerza float32 (default bf16/fp16 en CUDA)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    extract(
        stimuli_dir=args.stimuli_dir,
        out_path=args.out,
        model_id=args.model_id,
        batch_size=args.batch_size,
        limit=args.limit,
        device_str=args.device,
        force_fp32=args.fp32,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
