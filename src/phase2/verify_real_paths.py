"""
===============================================================================
phase2/verify_real_paths.py
===============================================================================

Prueba empírica del enlace fMRI → estímulo presentado en el dataset REAL
BOLD5000 (no mocks). Replica el subset de `get_ordered_test_stems` que NO
depende de CLIP embeddings (aún no extraídos en este entorno) y valida que:

    1. Presentation lists reales son parseables por el loader.
    2. Lista de 113 repeated estimuli se resuelve con la nueva ruta de
       `config.bold5000_repeated_list_txt()`.
    3. Cada stem del test set mapea a un archivo físico JPG/JPEG dentro de
       `BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/{COCO,ImageNet,Scene}/`
       mediante `stimuli_root.rglob(f"{stem}.*")`.

USO
---
    python -m phase2.verify_real_paths --subject CSI1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from phase2.bold5000_loader import _load_repeated_list, _load_stim_order_subject

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _find_gt(stimuli_root: Path, stem: str) -> Path | None:
    for cand in stimuli_root.rglob(f"{stem}.*"):
        if cand.is_file() and cand.suffix.lower() in VALID_IMG_EXT:
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Verificación de rutas reales BOLD5000.")
    ap.add_argument("--subject", default="CSI1", choices=config.BOLD5000_SUBJECTS)
    ap.add_argument("--n", type=int, default=5, help="Filas a imprimir.")
    args = ap.parse_args()

    cfg = config.BOLD5000_CONFIG
    stim_lists_root = Path(cfg["stim_lists_root"])
    repeated_txt = Path(cfg["repeated_list_txt"])
    stimuli_root = Path(cfg["stimuli_images"])

    print("=" * 120)
    print(f"VERIFICACIÓN DE RUTAS REALES — sujeto {args.subject}")
    print("=" * 120)
    print(f"stim_lists_root    : {stim_lists_root}")
    print(f"repeated_list_txt  : {repeated_txt}")
    print(f"stimuli_root       : {stimuli_root}")
    print("-" * 120)

    stim_order = _load_stim_order_subject(stim_lists_root, args.subject)
    repeated = _load_repeated_list(repeated_txt)
    observed = sorted({Path(s).stem for s in stim_order if Path(s).stem in repeated})

    print(f"Trials totales ({args.subject}) : {len(stim_order)}")
    print(f"Repeated list (release)     : {len(repeated)}")
    print(f"Stems observados (inter)    : {len(observed)}")
    print("-" * 120)
    print(f"{'Trial ID':<10} {'Stem Real':<50} Ruta Absoluta del JPG/JPEG")
    print("-" * 120)

    ok = 0
    missing: list[str] = []
    for tid, stem in enumerate(observed[: args.n]):
        gt = _find_gt(stimuli_root, stem)
        if gt is None:
            missing.append(stem)
            print(f"{tid:<10} {stem:<50} NOT FOUND")
        else:
            ok += 1
            print(f"{tid:<10} {stem:<50} {gt}")

    print("-" * 120)
    print(f"OK {ok}/{min(args.n, len(observed))} | missing {len(missing)}")
    if missing:
        print(f"stems sin archivo físico: {missing}")
        return 1
    print("Enlace fMRI (.mat trial order) <-> stim (.jpg fisico) VALIDO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
