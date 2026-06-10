"""
Tests del Ridge Estocástico (contribución central de la tesis).

Valida las ecuaciones 6.2-6.3 (ruido + renorm), la métrica de calibración
(pairwise accuracy) y que el barrido de σ elige señal. Sin GPU/disco.
"""

from __future__ import annotations

import numpy as np
import pytest

from phase2.adapter_ridge_stoch import (
    DEFAULT_SIGMAS,
    StochasticRidgeAdapter,
    calibrate_sigma,
    pairwise_accuracy,
    stochastic_transform,
)


# ---------------------------------------------------------------------------
# Transformación estocástica (eqs 6.2-6.3)
# ---------------------------------------------------------------------------

def test_stochastic_transform_renorm_is_unit():
    rng = np.random.default_rng(0)
    e = rng.standard_normal((10, 768)).astype(np.float32) * 7.0
    out = stochastic_transform(e, sigma=0.1, rng=np.random.default_rng(1), renorm=True)
    assert out.shape == (10, 768)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.ones(10), atol=1e-4)


def test_stochastic_transform_scale_applies_after_renorm():
    rng = np.random.default_rng(2)
    e = rng.standard_normal((8, 64)).astype(np.float32)
    out = stochastic_transform(e, sigma=0.1, rng=np.random.default_rng(3), renorm=True, scale=12.0)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.full(8, 12.0), atol=1e-3)


def test_stochastic_transform_sigma_zero_renorm_only():
    e = np.tile(np.arange(4, dtype=np.float32), (3, 1))  # (3,4)
    out = stochastic_transform(e, sigma=0.0, rng=np.random.default_rng(0), renorm=True)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# Pairwise accuracy
# ---------------------------------------------------------------------------

def test_pairwise_accuracy_identical_is_one():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((20, 32)).astype(np.float32)
    assert pairwise_accuracy(Y, Y) == pytest.approx(1.0)


def test_pairwise_accuracy_shuffled_near_chance():
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((64, 32)).astype(np.float32)
    perm = rng.permutation(64)
    acc = pairwise_accuracy(Y[perm], Y)
    assert 0.35 < acc < 0.65


# ---------------------------------------------------------------------------
# Calibración de σ
# ---------------------------------------------------------------------------

def test_calibrate_sigma_returns_valid_and_full_table():
    rng = np.random.default_rng(0)
    e_lin = rng.standard_normal((40, 32)).astype(np.float32)
    Y = rng.standard_normal((40, 32)).astype(np.float32)
    res = calibrate_sigma(e_lin, Y, sigmas=DEFAULT_SIGMAS, n_seeds=3, seed=0)
    assert res.best_sigma in DEFAULT_SIGMAS
    assert len(res.table) == len(DEFAULT_SIGMAS)
    assert 0.0 <= res.best_pairwise <= 1.0


def test_calibrate_sigma_prefers_low_noise_when_ridge_is_good():
    # Si el Ridge ya predice perfecto (e_lin == Y), menos ruido => mejor pairwise.
    rng = np.random.default_rng(7)
    Y = rng.standard_normal((50, 32)).astype(np.float32)
    res = calibrate_sigma(Y.copy(), Y, sigmas=(0.05, 0.8), n_seeds=5, seed=0)
    assert res.best_sigma == 0.05


# ---------------------------------------------------------------------------
# Adapter (envuelve un RidgeAdapter ajustado sobre split mock)
# ---------------------------------------------------------------------------

def test_adapter_predict_shape_on_mock_split():
    from phase2.adapter_ridge import RidgeAdapter
    from phase2.loader import load_split

    split = load_split(subject="CSI1", mode="mock",
                       loader_kwargs=dict(n_train=40, n_test=10, n_voxels=64, embed_dim=768, seed=0))
    ridge = RidgeAdapter(alpha=1e4).fit(split.betas_train, split.clip_train)
    adapter = StochasticRidgeAdapter(ridge, sigma=0.1, renorm=True)
    out = adapter.predict(split.betas_test, seed=0)
    assert out.shape == (10, 768)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.ones(10), atol=1e-4)
