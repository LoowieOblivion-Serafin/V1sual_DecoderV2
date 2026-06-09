"""
Tests del backbone MindEye y su pérdida híbrida (InfoNCE + MSE + Cosine).

Valida el CONTRATO (shapes, claves, finitud, gradiente) sin GPU. Es el núcleo
de la decodificación fMRI→CLIP: si esto se rompe, las reconstrucciones de la
Máquina B saldrán degeneradas.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from phase2.mindeye_models import MindEyeBackbone, MindEyeLoss


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

def test_backbone_forward_shape():
    B, V, D = 4, 128, 64
    net = MindEyeBackbone(in_voxels=V, out_dim=D, hidden_dim=96, n_blocks=2)
    out = net(torch.randn(B, V))
    assert out.shape == (B, D)
    assert torch.isfinite(out).all()


def test_backbone_rejects_bad_input_dim():
    net = MindEyeBackbone(in_voxels=32, out_dim=16, hidden_dim=48, n_blocks=1)
    with pytest.raises(ValueError):
        net(torch.randn(4, 31))          # última dim != in_voxels
    with pytest.raises(ValueError):
        net(torch.randn(32))             # no es 2D


def test_backbone_invalid_construction():
    with pytest.raises(ValueError):
        MindEyeBackbone(in_voxels=0, out_dim=16)
    with pytest.raises(ValueError):
        MindEyeBackbone(in_voxels=32, out_dim=0)
    with pytest.raises(ValueError):
        MindEyeBackbone(in_voxels=32, out_dim=16, n_blocks=0)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def test_loss_keys_and_scalar():
    loss_fn = MindEyeLoss()
    out = loss_fn(torch.randn(8, 32), torch.randn(8, 32))
    assert {"loss", "loss_nce", "loss_mse", "loss_cos"} <= set(out)
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])


def test_loss_perfect_alignment_low_nce():
    """Cuando pred == target, la InfoNCE debe ser baja (diagonal domina)."""
    loss_fn = MindEyeLoss()
    target = torch.randn(16, 64)
    aligned = loss_fn(target.clone(), target)["loss_nce"]
    shuffled = loss_fn(target[torch.randperm(16)], target)["loss_nce"]
    assert aligned < shuffled


def test_loss_backward_populates_grads():
    net = MindEyeBackbone(in_voxels=64, out_dim=32, hidden_dim=80, n_blocks=2)
    loss_fn = MindEyeLoss()
    pred = net(torch.randn(6, 64))
    out = loss_fn(pred, torch.randn(6, 32))
    out["loss"].backward()
    assert net.input_proj[0].weight.grad is not None
    assert loss_fn.logit_scale.grad is not None


def test_loss_logit_scale_clamped_safe():
    """logit_scale enorme no debe producir inf/nan tras el clamp+exp."""
    loss_fn = MindEyeLoss()
    with torch.no_grad():
        loss_fn.logit_scale.fill_(50.0)        # muy por encima de log(100)
    out = loss_fn(torch.randn(8, 32), torch.randn(8, 32))
    assert torch.isfinite(out["loss"])


def test_loss_rejects_shape_mismatch():
    loss_fn = MindEyeLoss()
    with pytest.raises(ValueError):
        loss_fn(torch.randn(4, 32), torch.randn(4, 16))
