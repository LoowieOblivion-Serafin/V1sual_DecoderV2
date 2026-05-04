"""
===============================================================================
phase2/visualize_results.py
===============================================================================

Utilidad CLI para cerrar el loop visual de la Fase 2 (tesis):
genera composiciones lado a lado [Ground Truth] vs [Reconstrucción SD 2.1]
para un sujeto BOLD5000 dado.

FLUJO
-----
1. Resuelve dinámicamente:
     - stimuli_root = config.BOLD5000_CONFIG['stimuli_images']
       (subdirs: COCO/ImageNet/Scene → se recorre con rglob).
     - recon_dir   = config.DATA_DIRS['output'] / {subject}
       (o el override --recon-dir).
2. Por cada PNG en `recon_dir` que matchee `--pattern` (default
   `{stem}_recon.png`), extrae el `stem` y busca el original
   `stimuli_root.rglob(f"{stem}.*")`.
3. Renderiza una figura de 2 paneles ("Ground Truth" vs "Reconstrucción")
   y la guarda en `--out-dir` (default: `recon_dir/comparisons`).

USO
---
    python -m phase2.visualize_results --subject CSI1
    python -m phase2.visualize_results --subject CSI1 --limit 10
    python -m phase2.visualize_results --subject CSI1 --pattern "{stem}_sd_unclip.png"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend headless (Máquina B sin display)
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

logger = logging.getLogger("phase2.visualize_results")

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Resolución de paths y stems
# ---------------------------------------------------------------------------

def _split_pattern(pattern: str) -> tuple[str, str]:
    """Descompone `{stem}_recon.png` en (prefix, suffix) alrededor de {stem}."""
    if "{stem}" not in pattern:
        raise ValueError(f"Pattern debe contener '{{stem}}': {pattern!r}")
    prefix, suffix = pattern.split("{stem}", 1)
    return prefix, suffix


def _iter_recon_stems(recon_dir: Path, pattern: str) -> list[tuple[str, Path]]:
    """Lista [(stem, ruta_recon)] para cada archivo que cuadra con el pattern."""
    prefix, suffix = _split_pattern(pattern)
    pairs: list[tuple[str, Path]] = []
    for f in sorted(recon_dir.iterdir()):
        if not f.is_file():
            continue
        name = f.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        stem = name[len(prefix): len(name) - len(suffix)] if suffix else name[len(prefix):]
        if stem:
            pairs.append((stem, f))
    return pairs


def _find_original(stimuli_root: Path, stem: str) -> Path | None:
    """rglob recursivo en COCO/ImageNet/Scene. Devuelve primer match con ext válida."""
    for candidate in stimuli_root.rglob(f"{stem}.*"):
        if candidate.is_file() and candidate.suffix.lower() in VALID_IMG_EXT:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _render_pair(
    gt_path: Path,
    recon_path: Path,
    out_path: Path,
    stem: str,
    dpi: int = 120,
) -> None:
    gt = Image.open(gt_path).convert("RGB")
    recon = Image.open(recon_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))
    axes[0].imshow(gt)
    axes[0].set_title("Ground Truth", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(recon)
    axes[1].set_title("Reconstrucción (SD 2.1 unCLIP)", fontsize=11)
    axes[1].axis("off")

    fig.suptitle(stem, fontsize=9, y=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def run(
    subject: str,
    recon_dir: Path,
    stimuli_root: Path,
    out_dir: Path,
    pattern: str,
    limit: int | None,
    dpi: int,
) -> tuple[int, int, int]:
    """Devuelve (ok, missing_original, failed)."""
    if not recon_dir.is_dir():
        raise FileNotFoundError(f"recon_dir no existe: {recon_dir}")
    if not stimuli_root.is_dir():
        raise FileNotFoundError(f"stimuli_root no existe: {stimuli_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _iter_recon_stems(recon_dir, pattern)
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        logger.warning(
            f"[{subject}] sin matches en {recon_dir} con pattern {pattern!r}"
        )
        return 0, 0, 0

    ok = missing = failed = 0
    for i, (stem, recon_path) in enumerate(pairs, 1):
        gt_path = _find_original(stimuli_root, stem)
        if gt_path is None:
            logger.warning(f"[{subject}] ({i}/{len(pairs)}) original no hallado: {stem}")
            missing += 1
            continue

        out_path = out_dir / f"{stem}_compare.png"
        try:
            _render_pair(gt_path, recon_path, out_path, stem, dpi=dpi)
            ok += 1
            logger.info(f"[{subject}] ({i}/{len(pairs)}) {stem} → {out_path.name}")
        except Exception as exc:
            failed += 1
            logger.error(f"[{subject}] ({i}/{len(pairs)}) fallo {stem}: {exc}")

    return ok, missing, failed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compara GT vs reconstrucción SD 2.1 unCLIP para un sujeto BOLD5000."
    )
    ap.add_argument("--subject", required=True,
                    choices=config.BOLD5000_SUBJECTS,
                    help="Sujeto BOLD5000 (CSI1..CSI4).")
    ap.add_argument("--pattern", default="{stem}_recon.png",
                    help="Pattern de filenames en recon-dir. Debe contener {stem}.")
    ap.add_argument("--recon-dir", type=Path, default=None,
                    help="Override del dir de reconstrucciones. "
                         "Default: config.DATA_DIRS['output']/{subject}")
    ap.add_argument("--stimuli-root", type=Path, default=None,
                    help="Override del dir raíz de estímulos. "
                         "Default: config.BOLD5000_CONFIG['stimuli_images']")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Dir de salida de composiciones. Default: recon-dir/comparisons")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dpi", type=int, default=120)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    stimuli_root = args.stimuli_root or config.BOLD5000_CONFIG["stimuli_images"]
    recon_dir = args.recon_dir or (config.DATA_DIRS["output"] / args.subject)
    out_dir = args.out_dir or (recon_dir / "comparisons")

    logger.info(f"subject={args.subject}")
    logger.info(f"stimuli_root={stimuli_root}")
    logger.info(f"recon_dir={recon_dir}")
    logger.info(f"out_dir={out_dir}")
    logger.info(f"pattern={args.pattern!r}")

    ok, missing, failed = run(
        subject=args.subject,
        recon_dir=Path(recon_dir),
        stimuli_root=Path(stimuli_root),
        out_dir=Path(out_dir),
        pattern=args.pattern,
        limit=args.limit,
        dpi=args.dpi,
    )

    logger.info(
        f"[{args.subject}] done — ok={ok} missing_gt={missing} failed={failed} "
        f"→ {out_dir}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
