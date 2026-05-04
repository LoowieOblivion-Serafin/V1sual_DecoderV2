"""
===============================================================================
phase2/loader.py — Facade único de carga de datos Fase 2
===============================================================================

Reemplaza al viejo `nsd_loader.py`. Despacha entre:

    - "bold5000" : datos reales BOLD5000 vía `bold5000_loader.load_bold5000_split`.
                   Paths se resuelven automáticamente desde `config.BOLD5000_CONFIG`
                   salvo override explícito en `loader_kwargs`.
    - "mock"     : datos sintéticos (`mock_data.make_mock_split`) para smoke tests
                   sin tocar disco.

No hay rama NSD (purgada con el pivote OpenNeuro). Toda la lógica de paths
vive en `config.py` — este módulo es puramente dispatcher + normalización al
dataclass `Split` compartido.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import config
from .mock_data import make_mock_split


# ---------------------------------------------------------------------------
# Dataclass shared por toda Fase 2
# ---------------------------------------------------------------------------

@dataclass
class Split:
    """Contrato de datos que consumen RidgeAdapter / LoRA adapter / trainer."""
    betas_train: np.ndarray      # (N_train, V_voxels)  float32
    betas_test:  np.ndarray      # (N_test,  V_voxels)  float32
    clip_train:  np.ndarray      # (N_train, 768)       float32
    clip_test:   np.ndarray      # (N_test,  768)       float32
    trial_ids_train: list[int]
    trial_ids_test:  list[int]
    subject: str


# ---------------------------------------------------------------------------
# Resolución de kwargs desde config.BOLD5000_CONFIG
# ---------------------------------------------------------------------------

def _bold5000_default_kwargs(subject: str) -> dict[str, Any]:
    """
    Extrae las rutas por defecto para `load_bold5000_split` a partir de
    `config.BOLD5000_CONFIG`. El caller (e.g. trainer) puede override campos
    concretos pasando `loader_kwargs`, con merge shallow (override gana).
    """
    cfg = config.BOLD5000_CONFIG
    rois_mat_fn = cfg["rois_mat"]               # callable(subject) -> Path
    return {
        "rois_mat":          rois_mat_fn(subject),
        "stim_lists_root":   cfg["stim_lists_root"],
        "repeated_list_txt": cfg["repeated_list_txt"],
        "clip_targets_pt":   cfg["clip_targets_pt"],
        "roi_names":         cfg["roi_set"],
    }


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def load_split(
    subject: str = "CSI1",
    mode: str = "bold5000",
    loader_kwargs: dict[str, Any] | None = None,
) -> Split:
    """
    Construye un `Split` entrenable.

    Args:
        subject: Uno de `config.BOLD5000_SUBJECTS` en modo 'bold5000'.
                 En modo 'mock' se propaga al `Split.subject` pero no afecta
                 la generación sintética.
        mode:    "bold5000" (datos reales, default) | "mock" (sanity sintético).
        loader_kwargs: overrides al loader subyacente.
            - bold5000: acepta keys rois_mat / stim_lists_root /
              repeated_list_txt / clip_targets_pt / roi_names / z_score.
              Missing keys se rellenan desde `config.BOLD5000_CONFIG`.
            - mock: kwargs de `make_mock_split` (n_train, n_test, n_voxels,
              embed_dim, snr, seed).

    Returns:
        Split con arrays float32 listos para `RidgeAdapter.fit`.

    Raises:
        ValueError: modo no soportado.
        FileNotFoundError / ValueError: propagados desde el loader concreto
            si faltan paths o shape mismatch.
    """
    loader_kwargs = dict(loader_kwargs or {})

    if mode == "mock":
        m = make_mock_split(**loader_kwargs)
        return Split(
            betas_train=m.betas_train,
            betas_test=m.betas_test,
            clip_train=m.clip_train,
            clip_test=m.clip_test,
            trial_ids_train=m.trial_ids_train,
            trial_ids_test=m.trial_ids_test,
            subject=subject,
        )

    if mode == "bold5000":
        # Import tardío: `bold5000_loader` importa scipy, innecesario en mock.
        from .bold5000_loader import load_bold5000_split

        defaults = _bold5000_default_kwargs(subject)
        defaults.update(loader_kwargs)          # overrides explícitos del caller
        return load_bold5000_split(subject=subject, **defaults)

    raise ValueError(f"mode debe ser 'bold5000' o 'mock'; recibido: {mode!r}")
