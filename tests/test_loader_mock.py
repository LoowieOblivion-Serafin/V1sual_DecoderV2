"""
Tests del facade de carga (`phase2.loader.load_split`) en modo 'mock'.

Garantiza el contrato `Split` que consumen el trainer y el evaluador, sin
tocar disco ni el dataset BOLD5000 real.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from phase2.loader import Split, load_split


def _small_split():
    return load_split(
        subject="CSI1",
        mode="mock",
        loader_kwargs=dict(n_train=20, n_test=8, n_voxels=64, embed_dim=768, snr=2.0, seed=0),
    )


def test_mock_split_contract():
    split = _small_split()
    assert isinstance(split, Split)
    assert split.subject == "CSI1"
    assert split.betas_train.shape == (20, 64)
    assert split.betas_test.shape == (8, 64)
    assert split.clip_train.shape == (20, 768)
    assert split.clip_test.shape == (8, 768)
    assert len(split.trial_ids_train) == 20
    assert len(split.trial_ids_test) == 8


def test_mock_split_dtypes_float32():
    split = _small_split()
    for arr in (split.betas_train, split.betas_test, split.clip_train, split.clip_test):
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()


def test_mock_split_voxel_dim_matches_betas_and_clip():
    split = _small_split()
    assert split.betas_train.shape[1] == split.betas_test.shape[1]
    assert split.clip_train.shape[1] == split.clip_test.shape[1]


def test_loader_rejects_unknown_mode():
    with pytest.raises(ValueError):
        load_split(subject="CSI1", mode="not_a_mode")
