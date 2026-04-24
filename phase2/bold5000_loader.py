"""
===============================================================================
phase2/bold5000_loader.py
===============================================================================

Loader REAL para BOLD5000 (pivote OpenNeuro confirmado). Produce un `Split`
(dataclass definido en `phase2/loader.py`) que consumen el `RidgeAdapter`,
`train_adapter.py` y en última instancia `phase2_run_sd.py`.

FUENTES DE DATOS
----------------
- Betas por trial (ROIs release, figshare):
    BOLD5000_ROIs/ROIs/CSI{N}/mat/CSI{N}_ROIs_TR34.mat
  Keys canónicas (según release oficial BOLD5000):
    LHLOC, RHLOC, LHPPA, RHPPA, LHEarlyVis, RHEarlyVis,
    LHRSC, RHRSC, LHOPA, RHOPA.
  Cada key es (N_trials, V_voxels_roi) en orden temporal de adquisición.

- Orden de estímulos presentados al sujeto (uno por trial):
    BOLD5000_Stimuli/Stimuli_Presentation_Lists/CSI{N}/CSI{N}_sess{MM}/
        CSI_sess{MM}_run{RR}.txt
  Sesiones sess01..sess15, runs run01..runNN. Cada .txt lista los estímulos
  presentados en ese run (un archivo por línea, extensión incluida:
  `.jpg` COCO, `.JPEG` ImageNet, `.jpg` Scene). El orden concatenado
  (session-major → run-major → trial-within-run) = orden temporal
  de adquisición = orden de filas del .mat.

- Lista de 113 estímulos repetidos (test set compartido):
    BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/repeated_stimuli_113_list.txt

- Targets CLIP ViT-L/14 (generados por `phase2/extract_vit_features.py`):
    phase2_outputs/clip_targets/bold5000_vitL14.pt
      {"filenames": [...], "embeddings": tensor (N, 768), "model_id": ..., "dim": 768}

SPLIT
-----
Test  = 113 estímulos repetidos, promediando sus ~4 repeticiones por sujeto.
Train = resto (~4803 trials por sujeto), 1 repetición → 1 trial.

No hay shuffle (mantenemos orden natural); z-score por vóxel usando
estadísticas SOLO de train, para evitar leakage al test shared.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .loader import Split

logger = logging.getLogger("phase2.bold5000_loader")

DEFAULT_ROI_SET: tuple[str, ...] = (
    "LHLOC", "RHLOC",
    "LHPPA", "RHPPA",
    "LHEarlyVis", "RHEarlyVis",
    "LHRSC", "RHRSC",
    "LHOPA", "RHOPA",
)


# ---------------------------------------------------------------------------
# Lectores de archivos brutos
# ---------------------------------------------------------------------------

def _load_roi_mat(mat_path: Path, roi_names: tuple[str, ...]) -> np.ndarray:
    """
    Concatena los ROIs pedidos en (N_trials, V_total).

    `scipy.io.loadmat(..., squeeze_me=True)` devuelve arrays 2D por ROI.
    Trial order = orden temporal de adquisición (session-major, run-major).
    """
    if not mat_path.exists():
        raise FileNotFoundError(f"ROIs .mat no encontrado: {mat_path}")

    mat = loadmat(str(mat_path), squeeze_me=True)
    missing = [r for r in roi_names if r not in mat]
    if missing:
        available = sorted(k for k in mat.keys() if not k.startswith("__"))
        raise KeyError(
            f"ROIs faltantes en {mat_path.name}: {missing}. "
            f"Disponibles: {available}"
        )

    blocks = []
    for r in roi_names:
        arr = np.asarray(mat[r])
        if arr.ndim != 2:
            raise ValueError(f"ROI {r!r} tiene ndim={arr.ndim}, esperado 2")
        blocks.append(arr.astype(np.float32))

    n_trials = {b.shape[0] for b in blocks}
    if len(n_trials) != 1:
        raise ValueError(f"ROIs con N_trials inconsistente: {n_trials}")

    return np.concatenate(blocks, axis=1)


def _load_stim_order_subject(stim_lists_root: Path, subject: str) -> list[str]:
    """
    Recorre BOLD5000_Stimuli/Stimuli_Presentation_Lists/{subject}/ y concatena
    todos los runs de todas las sesiones en orden `sess01..sess15` →
    `run01..runNN`. Devuelve la lista plana de filenames (1 por trial).

    Asunción verificada contra el release: sort alfabético de los .txt por
    sesión y por run reproduce el orden temporal del .mat. Si el chequeo de
    longitud contra el .mat falla, `load_bold5000_split` aborta con mensaje
    explícito (no silenciar).
    """
    subj_dir = stim_lists_root / subject
    if not subj_dir.is_dir():
        raise FileNotFoundError(f"Directorio de stim lists no existe: {subj_dir}")

    session_dirs = sorted(
        p for p in subj_dir.iterdir()
        if p.is_dir() and p.name.startswith(f"{subject}_sess")
    )
    if not session_dirs:
        raise RuntimeError(f"Ninguna sesión hallada en {subj_dir}")

    all_stims: list[str] = []
    for sess_dir in session_dirs:
        run_files = sorted(sess_dir.glob("*_run*.txt"))
        if not run_files:
            raise RuntimeError(f"Sesión sin runs .txt: {sess_dir}")
        for run_txt in run_files:
            with run_txt.open(encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
            all_stims.extend(lines)

    logger.info(
        f"[{subject}] stim order: {len(all_stims)} trials en "
        f"{len(session_dirs)} sesiones ({session_dirs[0].name}..{session_dirs[-1].name})"
    )
    return all_stims


def _load_repeated_list(repeated_txt: Path) -> set[str]:
    """
    Lee `repeated_stimuli_113_list.txt`. Devuelve el set de **stems** (sin ext)
    para hacer match directo contra los embeddings CLIP.
    """
    if not repeated_txt.exists():
        raise FileNotFoundError(f"Lista de repeated no encontrada: {repeated_txt}")
    stems: set[str] = set()
    with repeated_txt.open(encoding="utf-8") as fh:
        for ln in fh:
            s = ln.strip()
            if s:
                stems.add(Path(s).stem)
    if len(stems) != 113:
        logger.warning(f"Esperaba 113 repeated, obtuve {len(stems)}")
    return stems


def _build_clip_lookup(clip_pt_path: Path) -> dict[str, np.ndarray]:
    """
    Indexa embeddings CLIP ViT-L/14 por **stem** del filename. La clave es
    compatible con el stim order y con la lista de repeated.
    """
    import torch
    if not clip_pt_path.exists():
        raise FileNotFoundError(f"CLIP targets no encontrados: {clip_pt_path}")
    blob = torch.load(clip_pt_path, map_location="cpu")
    if "filenames" not in blob or "embeddings" not in blob:
        raise ValueError(
            f"{clip_pt_path} no tiene las keys esperadas "
            f"('filenames', 'embeddings'). Recibido: {list(blob.keys())}"
        )
    emb = blob["embeddings"].numpy().astype(np.float32)
    if emb.ndim != 2 or emb.shape[1] != 768:
        raise ValueError(f"CLIP targets shape inválido: {emb.shape}, esperado (N, 768)")
    return {Path(fn).stem: emb[i] for i, fn in enumerate(blob["filenames"])}


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def load_bold5000_split(
    subject: str,                         # "CSI1" | "CSI2" | "CSI3" | "CSI4"
    rois_mat: Path,                       # BOLD5000_ROIs/ROIs/{subject}/mat/{subject}_ROIs_TR34.mat
    stim_lists_root: Path,                # BOLD5000_Stimuli/Stimuli_Presentation_Lists
    repeated_list_txt: Path,              # BOLD5000_Stimuli/.../repeated_stimuli_113_list.txt
    clip_targets_pt: Path,                # phase2_outputs/clip_targets/bold5000_vitL14.pt
    roi_names: tuple[str, ...] = DEFAULT_ROI_SET,
    z_score: bool = True,
) -> Split:
    """
    Construye el split BOLD5000 train/test alineado con CLIP ViT-L/14.

    Pasos:
        1. Lee betas (N_trials, V_total) desde el .mat de ROIs.
        2. Reconstruye el orden de estímulos recorriendo sesiones y runs.
        3. Separa trials en TRAIN (no repetidos) y TEST (repetidos).
        4. Promedia repeticiones del test set por stem.
        5. z-score opcional por vóxel, usando SOLO train (evita leakage).

    Contratos:
        - len(stim_order) == betas.shape[0] (chequeo explícito).
        - Todo stem en stim_order debe existir en clip_lut, salvo los skip
          controlados (log.warning). Los trials sin embedding se descartan.
    """
    betas = _load_roi_mat(rois_mat, roi_names)
    stim_order = _load_stim_order_subject(stim_lists_root, subject)

    if betas.shape[0] != len(stim_order):
        raise ValueError(
            f"Mismatch N_trials: betas={betas.shape[0]} vs stim_order={len(stim_order)} "
            f"para {subject}. Revisa que el .mat sea TR34 y que todas las sesiones "
            f"estén presentes en {stim_lists_root / subject}."
        )

    repeated_stems = _load_repeated_list(repeated_list_txt)
    clip_lut = _build_clip_lookup(clip_targets_pt)

    # Particionar trials
    test_idx_by_stem: dict[str, list[int]] = {}
    train_idx: list[int] = []
    missing_in_clip = 0

    for i, stim in enumerate(stim_order):
        stem = Path(stim).stem
        if stem not in clip_lut:
            missing_in_clip += 1
            continue
        if stem in repeated_stems:
            test_idx_by_stem.setdefault(stem, []).append(i)
        else:
            train_idx.append(i)

    if missing_in_clip:
        logger.warning(
            f"[{subject}] {missing_in_clip} trials sin embedding CLIP "
            f"(posible desajuste entre --stimuli-dir y presentation lists)"
        )

    # TRAIN: 1 trial → 1 target
    X_train = betas[train_idx]
    Y_train = np.stack([clip_lut[Path(stim_order[i]).stem] for i in train_idx])
    ids_train = list(range(len(train_idx)))

    # TEST: media de repeticiones por stem
    X_test_list, Y_test_list, stems_test = [], [], []
    for stem in sorted(test_idx_by_stem):
        reps_idx = test_idx_by_stem[stem]
        X_test_list.append(betas[reps_idx].mean(axis=0))
        Y_test_list.append(clip_lut[stem])
        stems_test.append(stem)

    if not X_test_list:
        raise RuntimeError(
            f"[{subject}] test set vacío. ¿Está la lista repeated OK y los "
            f"stims de stim_order coinciden con ella? "
            f"repeated n={len(repeated_stems)}"
        )

    X_test = np.stack(X_test_list).astype(np.float32)
    Y_test = np.stack(Y_test_list).astype(np.float32)
    ids_test = list(range(len(stems_test)))

    # z-score por vóxel (estadísticas de train únicamente)
    if z_score:
        mu = X_train.mean(axis=0, keepdims=True)
        sd = X_train.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-6, 1.0, sd)  # vóxeles planos no introducen NaN
        X_train = (X_train - mu) / sd
        X_test = (X_test - mu) / sd

    logger.info(
        f"[{subject}] train={X_train.shape} test={X_test.shape} "
        f"ROIs={len(roi_names)} V={X_train.shape[1]} "
        f"reps_test={ {s: len(test_idx_by_stem[s]) for s in stems_test[:3]} }..."
    )

    return Split(
        betas_train=X_train.astype(np.float32),
        betas_test=X_test.astype(np.float32),
        clip_train=Y_train.astype(np.float32),
        clip_test=Y_test.astype(np.float32),
        trial_ids_train=ids_train,
        trial_ids_test=ids_test,
        subject=subject,
    )
