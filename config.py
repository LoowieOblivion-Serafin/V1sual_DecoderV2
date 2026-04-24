"""
===============================================================================
CONFIGURACIÓN DEL PROYECTO — Fase 2 BOLD5000 + SD 2.1 unCLIP
===============================================================================

Centraliza paths, parámetros de inferencia y procesamiento para el pipeline
post-pivote BOLD5000 (OpenNeuro). La rama VGG+VQGAN (Fase 1) fue purgada;
toda referencia se encuentra en MIGRATION.md como contexto histórico.

PORTABILIDAD DE RUTAS
---------------------
El código se desarrolla en Windows (Máquina A) y se ejecuta en Linux/Windows
sobre RTX 4070 Ti (Máquina B). Todas las rutas base se resuelven:

    1. Variable de entorno `ACECOM_<KEY>` si está definida.
    2. Fallback relativo a `Path(__file__).resolve().parent` (raíz del repo).

Ninguna ruta absoluta local está hardcodeada. Para sobrescribir en Máquina B:

    export ACECOM_BOLD5000_ROIS_ROOT=/mnt/data/BOLD5000_ROIs
    export ACECOM_BOLD5000_STIMULI_ROOT=/mnt/data/BOLD5000_Stimuli
    export ACECOM_OUTPUT_ROOT=/mnt/scratch/acecom_out
    export ACECOM_HF_CACHE=/mnt/models/hf
"""

from __future__ import annotations

import os
from pathlib import Path

# ============================================================================
# RAÍZ DEL PROYECTO
# ============================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent


def _env_path(key: str, default: Path) -> Path:
    """
    Resuelve una ruta desde variable de entorno `ACECOM_<key>` con fallback
    relativo al repositorio. Convierte siempre a `Path` absoluto.
    """
    raw = os.getenv(f"ACECOM_{key}")
    return Path(raw).expanduser().resolve() if raw else default.resolve()


# ============================================================================
# RUTAS DE DATOS (portables)
# ============================================================================

DATA_DIRS: dict[str, Path] = {
    # Raíces top-level (BOLD5000 release oficial)
    "bold5000_rois":    _env_path("BOLD5000_ROIS_ROOT",    PROJECT_ROOT / "BOLD5000_ROIs"),
    "bold5000_stimuli": _env_path("BOLD5000_STIMULI_ROOT", PROJECT_ROOT / "BOLD5000_Stimuli"),
    "bold5000_bids":    _env_path("BOLD5000_BIDS_ROOT",    PROJECT_ROOT / "data_bold5000"),

    # Artefactos generados por el pipeline (adapter, embeddings, etc.)
    "phase2_outputs":   _env_path("PHASE2_OUTPUTS",        PROJECT_ROOT / "phase2_outputs"),

    # Reconstrucciones SD 2.1 unCLIP (dumps PNG crudos del orquestador)
    "output":           _env_path("OUTPUT_ROOT",           PROJECT_ROOT / "output_sd_reconstructions"),

    # Evaluador visual (collages GT vs reconstrucción, grids y reports)
    "eval_output":      _env_path("EVAL_OUTPUT",           PROJECT_ROOT / "output_reconstructions_sd21"),

    # Cache local de Hugging Face (modelos descargados)
    "models_hf":        _env_path("HF_CACHE",              PROJECT_ROOT / "models_hf"),
}


# ============================================================================
# BOLD5000 — única fuente de verdad (pivote OpenNeuro)
# ============================================================================

# Sujetos disponibles en BOLD5000 ROIs release (figshare).
BOLD5000_SUBJECTS: tuple[str, ...] = ("CSI1", "CSI2", "CSI3", "CSI4")

# TR usado por la release ROIs. Las opciones son TR1..TR5 y TR34.
# TR34 = promedio de TR3+TR4 (pico HRF). Es la recomendada para decoding.
BOLD5000_TR_KEY: str = "TR34"

# ROIs a concatenar para construir X (vóxeles visuales).
# Orden canónico; load_bold5000_split valida shape por cada key.
BOLD5000_ROI_SET: tuple[str, ...] = (
    "LHLOC", "RHLOC",
    "LHPPA", "RHPPA",
    "LHEarlyVis", "RHEarlyVis",
    "LHRSC", "RHRSC",
    "LHOPA", "RHOPA",
)


def bold5000_rois_mat(subject: str) -> Path:
    """
    Ruta al .mat de betas por trial para `subject`.

    Ej.: BOLD5000_ROIs/ROIs/CSI1/mat/CSI1_ROIs_TR34.mat
    """
    if subject not in BOLD5000_SUBJECTS:
        raise ValueError(f"Sujeto inválido: {subject!r}. Usa uno de {BOLD5000_SUBJECTS}")
    return (
        DATA_DIRS["bold5000_rois"]
        / "ROIs" / subject / "mat"
        / f"{subject}_ROIs_{BOLD5000_TR_KEY}.mat"
    )


def bold5000_stim_lists_root() -> Path:
    """
    Directorio raíz con las listas de presentación por sesión/run.

    Estructura: BOLD5000_Stimuli/Stimuli_Presentation_Lists/CSI{N}/CSI{N}_sess{MM}/CSI_sess{MM}_run{RR}.txt
    """
    return DATA_DIRS["bold5000_stimuli"] / "Stimuli_Presentation_Lists"


def bold5000_repeated_list_txt() -> Path:
    """
    Lista de 113 estímulos repetidos (test set compartido).

    Release oficial: vive un nivel arriba de Presented_Stimuli, directamente
    bajo Scene_Stimuli/. Verificado contra figshare release.
    """
    return (
        DATA_DIRS["bold5000_stimuli"]
        / "Scene_Stimuli"
        / "repeated_stimuli_113_list.txt"
    )


def bold5000_stimuli_images_root() -> Path:
    """
    Directorio raíz con las imágenes presentadas (subdirs COCO/ImageNet/Scene).
    Lo consume `extract_vit_features.py` vía `--stimuli-dir`.
    """
    return DATA_DIRS["bold5000_stimuli"] / "Scene_Stimuli" / "Presented_Stimuli"


def bold5000_clip_targets_pt() -> Path:
    """
    Ruta estándar del .pt de embeddings CLIP ViT-L/14 generado por
    `phase2/extract_vit_features.py`.
    """
    return DATA_DIRS["phase2_outputs"] / "clip_targets" / "bold5000_vitL14.pt"


BOLD5000_CONFIG: dict[str, object] = {
    "subjects":          BOLD5000_SUBJECTS,
    "tr_key":            BOLD5000_TR_KEY,
    "roi_set":           BOLD5000_ROI_SET,
    "rois_mat":          bold5000_rois_mat,              # callable(subject) -> Path
    "stim_lists_root":   bold5000_stim_lists_root(),     # Path
    "repeated_list_txt": bold5000_repeated_list_txt(),   # Path
    "stimuli_images":    bold5000_stimuli_images_root(), # Path
    "clip_targets_pt":   bold5000_clip_targets_pt(),     # Path
}


# ============================================================================
# SD 2.1 unCLIP
# ============================================================================

SD_CONFIG: dict[str, object] = {
    "repo_id":             "diffusers/stable-diffusion-2-1-unclip-i2i-l",
    "clip_target_repo":    "openai/clip-vit-large-patch14",
    "embedding_dim":       768,
    "num_inference_steps": 25,
    "guidance_scale":      10.0,
    "noise_level":         0,
    "seed":                42,
    "image_size":          768,
}


# ============================================================================
# PROCESAMIENTO
# ============================================================================

PROCESSING_CONFIG: dict[str, object] = {
    "ignore_hidden_files": True,
    "save_format":         "png",
}


# ============================================================================
# HARDWARE
# ============================================================================

HARDWARE_CONFIG: dict[str, object] = {
    "force_cpu":       False,
    "gpu_id":          0,
    "use_bf16":        True,
    "enable_xformers": False,
}


# ============================================================================
# EVALUACIÓN
# ============================================================================

EVALUATION_CONFIG: dict[str, object] = {
    "metrics":     ["ssim", "pixcorr", "psnr", "lpips", "clip", "pairwise"],
    "export_csv":  True,
    "export_html": True,
}


# ============================================================================
# UTILIDADES
# ============================================================================

def filter_valid_files(file_list, ignore_hidden: bool = True, required_substring: str | None = None):
    valid = []
    for fname in file_list:
        if ignore_hidden and fname.startswith("._"):
            continue
        if required_substring and required_substring not in fname:
            continue
        valid.append(fname)
    return valid


def print_config() -> None:
    print("=" * 72)
    print("CONFIGURACIÓN — Fase 2 BOLD5000 + SD 2.1 unCLIP")
    print("=" * 72)
    print(f"PROJECT_ROOT:        {PROJECT_ROOT}")
    print(f"BOLD5000 ROIs root:  {DATA_DIRS['bold5000_rois']}")
    print(f"BOLD5000 Stimuli:    {DATA_DIRS['bold5000_stimuli']}")
    print(f"BOLD5000 BIDS:       {DATA_DIRS['bold5000_bids']}")
    print(f"Phase2 outputs:      {DATA_DIRS['phase2_outputs']}")
    print(f"Reconstructions:     {DATA_DIRS['output']}")
    print(f"Eval output:         {DATA_DIRS['eval_output']}")
    print(f"HF cache:            {DATA_DIRS['models_hf']}")
    print("-" * 72)
    print(f"Subjects:            {BOLD5000_SUBJECTS}")
    print(f"TR key:              {BOLD5000_TR_KEY}")
    print(f"ROI set ({len(BOLD5000_ROI_SET)}):      {BOLD5000_ROI_SET}")
    print(f"rois_mat(CSI1):      {bold5000_rois_mat('CSI1')}")
    print(f"stim_lists_root:     {bold5000_stim_lists_root()}")
    print(f"repeated_list_txt:   {bold5000_repeated_list_txt()}")
    print(f"stimuli_images:      {bold5000_stimuli_images_root()}")
    print(f"clip_targets_pt:     {bold5000_clip_targets_pt()}")
    print("-" * 72)
    print(f"SD repo:             {SD_CONFIG['repo_id']}")
    print(f"Device:              {'CPU' if HARDWARE_CONFIG['force_cpu'] else 'GPU'}")
    print("=" * 72)


if __name__ == "__main__":
    print_config()
