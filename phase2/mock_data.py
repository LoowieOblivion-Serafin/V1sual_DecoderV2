"""
Generador sintético NSD-like para validar el pipeline antes de tener acceso real.

Reproduce shapes y orden de magnitud esperados:
    - Betas fMRI:    [N_trials, V_voxels]   con V≈10000 (máscara nsdgeneral)
    - CLIP target:   [N_trials, 768]        ViT-L/14 embedding space
    - Train: ~9000 trials únicos (con repeticiones promediadas).
    - Test:  ~1000 trials shared1000 (3 reps promediadas).

La señal sintética es una mezcla lineal con ruido gaussiano: fija una matriz
W_true [V, 768] aleatoria, genera betas ~ N(0, 1), CLIP_target = betas @ W_true
+ ruido. Esto garantiza que un Ridge bien regularizado debe recuperar señal
(score R² > 0 en test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MockNSDSplit:
    betas_train: np.ndarray
    betas_test: np.ndarray
    clip_train: np.ndarray
    clip_test: np.ndarray
    trial_ids_train: list[int]
    trial_ids_test: list[int]


def make_mock_split(
    n_train: int = 9000,
    n_test: int = 1000,
    n_voxels: int = 10000,
    embed_dim: int = 768,
    snr: float = 0.5,
    seed: int = 42,
) -> MockNSDSplit:
    """
    snr: razón señal/ruido en CLIP target. snr=0.5 simula NSD realista
    (mucho ruido fisiológico). snr=2.0 = caso optimista para sanity check.
    """
    rng = np.random.default_rng(seed)

    W_true = rng.standard_normal((n_voxels, embed_dim)).astype(np.float32) / np.sqrt(n_voxels)

    betas_train = rng.standard_normal((n_train, n_voxels)).astype(np.float32)
    betas_test = rng.standard_normal((n_test, n_voxels)).astype(np.float32)

    clip_train_clean = betas_train @ W_true
    clip_test_clean = betas_test @ W_true

    noise_std_train = clip_train_clean.std() / snr
    noise_std_test = clip_test_clean.std() / snr

    clip_train = clip_train_clean + rng.standard_normal(clip_train_clean.shape).astype(np.float32) * noise_std_train
    clip_test = clip_test_clean + rng.standard_normal(clip_test_clean.shape).astype(np.float32) * noise_std_test

    trial_ids_train = list(range(n_train))
    trial_ids_test = list(range(n_train, n_train + n_test))

    return MockNSDSplit(
        betas_train=betas_train,
        betas_test=betas_test,
        clip_train=clip_train,
        clip_test=clip_test,
        trial_ids_train=trial_ids_train,
        trial_ids_test=trial_ids_test,
    )
