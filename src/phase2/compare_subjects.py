"""
===============================================================================
phase2/compare_subjects.py — Comparador multi-sujeto en una sola muestra
===============================================================================

Cierra el loop cualitativo CROSS-SUBJECT de la tesis: para un mismo estímulo
(de los 113 repeated, compartidos por los 4 sujetos BOLD5000) renderiza en una
sola figura el Ground Truth junto a la reconstrucción de CADA individuo:

    [ GT | CSI1 | CSI2 | CSI3 | CSI4 ]

Esto permite comparar con detalle cuánta información visual recupera cada
cerebro del MISMO estímulo, algo que el evaluador por-sujeto
(`visual_evaluator.py`) no muestra porque genera grids GT|Recon de un único
sujeto.

ENTRADA
-------
Lee las reconstrucciones que escribió `visual_evaluator.py`:

    {eval_dir}/{subject}/reconstructions/{stem}_recon.png

El Ground Truth se resuelve por `stem` recorriendo el árbol de estímulos
BOLD5000 (subdirs COCO/ImageNet/Scene), igual que el resto del pipeline.

MODOS
-----
1. Muestra agregada (default): una figura con N filas (estímulos) × (1+S)
   columnas. Ideal para un panel de resultados en la tesis.

       python -m phase2.compare_subjects --limit 8

2. Una sola muestra en detalle: un estímulo concreto, fila única, alta
   resolución. "Los cuatro individuos en una sola muestra".

       python -m phase2.compare_subjects --stem desertvegetation3 --dpi 200

PORTABILIDAD
------------
Sin rutas hardcodeadas: todo se deriva de `config` y de las variables
`ACECOM_EVAL_OUTPUT` / `ACECOM_BOLD5000_STIMULI_ROOT`. Sin dependencia de
torch/diffusers — corre en Máquina A (dev) o B (post git-pull) con sólo
matplotlib + PIL.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend headless (Máquina B sin display)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

logger = logging.getLogger("phase2.compare_subjects")

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RECON_SUBDIR = "reconstructions"
RECON_SUFFIX = "_recon.png"


# ---------------------------------------------------------------------------
# Resolución de paths
# ---------------------------------------------------------------------------

def _subject_recon_dir(eval_dir: Path, subject: str) -> Path:
    return eval_dir / subject / RECON_SUBDIR


def _recon_path(eval_dir: Path, subject: str, stem: str) -> Path:
    return _subject_recon_dir(eval_dir, subject) / f"{stem}{RECON_SUFFIX}"


def list_subject_stems(eval_dir: Path, subject: str) -> set[str]:
    """Stems reconstruidos disponibles para `subject` (set por intersección)."""
    rdir = _subject_recon_dir(eval_dir, subject)
    if not rdir.is_dir():
        logger.warning("[%s] sin dir de reconstrucciones: %s", subject, rdir)
        return set()
    stems: set[str] = set()
    for png in rdir.glob(f"*{RECON_SUFFIX}"):
        stems.add(png.name[: -len(RECON_SUFFIX)])
    return stems


def common_stems(eval_dir: Path, subjects: list[str]) -> list[str]:
    """
    Intersección de stems reconstruidos por TODOS los sujetos pedidos.

    Como los 113 repeated son compartidos, la intersección suele ser el test
    set completo. Devuelve la lista ordenada para reproducibilidad.
    """
    sets = [list_subject_stems(eval_dir, s) for s in subjects]
    sets = [s for s in sets if s]
    if not sets:
        return []
    inter = set.intersection(*sets)
    return sorted(inter)


def find_ground_truth(stimuli_root: Path, stem: str) -> Path | None:
    """rglob recursivo; primer match con extensión válida."""
    for cand in stimuli_root.rglob(f"{stem}.*"):
        if cand.is_file() and cand.suffix.lower() in VALID_IMG_EXT:
            return cand
    return None


# ---------------------------------------------------------------------------
# Carga de celdas (imagen o placeholder)
# ---------------------------------------------------------------------------

def _load_img(path: Path | None, size: int) -> np.ndarray:
    """Carga RGB redimensionada; si falta, devuelve placeholder gris con cruz."""
    if path is not None and path.is_file():
        return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC))
    ph = np.full((size, size, 3), 210, dtype=np.uint8)
    ph[size // 2 - 1: size // 2 + 1, :, :] = 150
    ph[:, size // 2 - 1: size // 2 + 1, :] = 150
    return ph


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_comparison(
    stems: list[str],
    eval_dir: Path,
    stimuli_root: Path,
    subjects: list[str],
    out_path: Path,
    cell_px: int,
    dpi: int,
) -> dict:
    """
    Figura (len(stems)) filas × (1 + len(subjects)) columnas:
        col 0 = GT, col j = recon del sujeto j.
    """
    if not stems:
        raise ValueError("Lista de stems vacía: ¿corriste visual_evaluator para todos los sujetos?")

    n_rows = len(stems)
    n_cols = 1 + len(subjects)
    col_titles = ["Estímulo (GT)"] + list(subjects)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.2 * n_cols, 2.2 * n_rows),
        squeeze=False,
    )

    n_missing = 0
    for i, stem in enumerate(stems):
        gt_path = find_ground_truth(stimuli_root, stem)
        cells = [gt_path] + [_recon_path(eval_dir, s, stem) for s in subjects]
        for j, cell_path in enumerate(cells):
            ax = axes[i][j]
            exists = cell_path is not None and cell_path.is_file()
            if j > 0 and not exists:
                n_missing += 1
            ax.imshow(_load_img(cell_path, cell_px))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(col_titles[j], fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(stem, fontsize=7, rotation=0, ha="right", va="center", labelpad=28)

    fig.suptitle(
        f"Comparación cross-subject — {len(subjects)} individuos · {n_rows} estímulos",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "out": str(out_path),
        "n_stems": n_rows,
        "n_subjects": len(subjects),
        "missing_cells": n_missing,
    }


# ---------------------------------------------------------------------------
# Selección de stems
# ---------------------------------------------------------------------------

def select_stems(
    eval_dir: Path,
    subjects: list[str],
    explicit: list[str] | None,
    limit: int | None,
    shuffle: bool,
    seed: int,
) -> list[str]:
    if explicit:
        return explicit
    stems = common_stems(eval_dir, subjects)
    if not stems:
        return []
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(stems)
    if limit is not None:
        stems = stems[:limit]
    return stems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compara la reconstrucción de los 4 sujetos BOLD5000 sobre el mismo estímulo."
    )
    ap.add_argument("--subjects", nargs="+", default=list(config.BOLD5000_SUBJECTS),
                    choices=list(config.BOLD5000_SUBJECTS),
                    help="Sujetos a comparar (default: los 4).")
    ap.add_argument("--eval-dir", type=Path, default=None,
                    help="Raíz de reconstrucciones. Default: config.DATA_DIRS['eval_output'].")
    ap.add_argument("--stimuli-root", type=Path, default=None,
                    help="Raíz de estímulos GT. Default: config.BOLD5000_CONFIG['stimuli_images'].")
    ap.add_argument("--out", type=Path, default=None,
                    help="Ruta del PNG de salida. Default: {eval_dir}/comparacion_4sujetos.png")
    ap.add_argument("--stem", nargs="+", default=None,
                    help="Uno o más stems concretos (modo 'una sola muestra' en detalle).")
    ap.add_argument("--limit", type=int, default=10,
                    help="Máx. de estímulos en modo agregado (ignorado si se pasa --stem).")
    ap.add_argument("--shuffle", action="store_true",
                    help="Baraja los stems comunes antes de aplicar --limit.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cell-px", type=int, default=256, help="Resolución por celda.")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    eval_dir = args.eval_dir or config.DATA_DIRS["eval_output"]
    stimuli_root = args.stimuli_root or config.BOLD5000_CONFIG["stimuli_images"]
    out_path = args.out or (eval_dir / "comparacion_4sujetos.png")

    logger.info("subjects     = %s", args.subjects)
    logger.info("eval_dir     = %s", eval_dir)
    logger.info("stimuli_root = %s", stimuli_root)

    stems = select_stems(
        eval_dir=Path(eval_dir),
        subjects=args.subjects,
        explicit=args.stem,
        limit=None if args.stem else args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not stems:
        logger.error(
            "Sin estímulos comunes a %s en %s. "
            "Corre `visual_evaluator` para cada sujeto antes de comparar.",
            args.subjects, eval_dir,
        )
        return 1

    summary = render_comparison(
        stems=stems,
        eval_dir=Path(eval_dir),
        stimuli_root=Path(stimuli_root),
        subjects=args.subjects,
        out_path=Path(out_path),
        cell_px=args.cell_px,
        dpi=args.dpi,
    )
    logger.info("done — %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
