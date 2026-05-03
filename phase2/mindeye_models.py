"""
===============================================================================
phase2/mindeye_models.py — MindEye-style fMRI → CLIP backbone + loss
===============================================================================

Reemplaza el `RidgeAdapter` lineal (Takagi-Nishimoto baseline) por una
arquitectura profunda residual entrenada con InfoNCE bidireccional, inspirada
en MindEye (Scotti et al., MedARC-AI, 2023/2024).

Motivación
----------
Ridge resuelve `Y = X·W` en forma cerrada y captura sólo correlaciones
lineales fMRI↔CLIP. Esto colapsa la geometría del manifold visual: SD 2.1
unCLIP recibe vectores con poca variabilidad semántica y degenera en texto /
alucinaciones genéricas. Una MLP residual con normalización y aprendizaje
contrastivo aprende la métrica del espacio CLIP en vez de minimizar MSE
elemento a elemento, lo que preserva direcciones discriminativas entre
clases visualmente cercanas.

Contrato (ver `phase2/INTERFACE.md`)
------------------------------------
- `MindEyeBackbone(in_voxels, out_dim, ...)` con forward `(B, in_voxels) → (B, out_dim)`.
- `MindEyeLoss()` con forward `(pred, target) → {loss, loss_nce, loss_mse, loss_cos}`.
- bf16-safe: sin `exp()` sobre valores no acotados, sin divisiones por
  normas crudas (siempre con epsilon vía `F.normalize`), sin softmax manual.

Referencias
-----------
- Scotti et al., 2023, "Reconstructing the Mind's Eye" (NeurIPS).
- Radford et al., 2021, "CLIP" — patrón `logit_scale = log(1/τ)` learnable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Bloque residual MLP
# =============================================================================

class _ResidualMLPBlock(nn.Module):
    """
    Bloque residual pre-norm para MLP de gran ancho.

    Estructura:
        x -> LayerNorm -> Linear -> GELU -> Dropout -> Linear -> + x

    Nota
    ----
    Pre-norm (LN antes de la primera Linear) facilita el flujo de gradiente
    en redes profundas y es estable en bf16. La conexión residual mantiene
    la dimensión `hidden_dim` constante en todo el stack.

    Shapes
    ------
    in:  (B, hidden_dim)
    out: (B, hidden_dim)
    """

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


# =============================================================================
# Backbone
# =============================================================================

class MindEyeBackbone(nn.Module):
    """
    MLP residual fMRI (vóxeles) → embedding CLIP.

    Arquitectura
    ------------
        Linear(in_voxels → hidden) → LayerNorm → GELU
        ├── ResidualMLPBlock × n_blocks
        └── LayerNorm → Linear(hidden → out_dim)

    La proyección final es lineal pura (sin activación) para no sesgar
    direcciones del espacio CLIP target.

    Parameters
    ----------
    in_voxels : int
        Número de vóxeles concatenados de los ROIs (dinámico por sujeto).
    out_dim : int
        Dimensión del embedding objetivo. Leer desde
        `config.SD_CONFIG["embedding_dim"]` (768 para CLIP ViT-L/14).
    hidden_dim : int, default 4096
        Ancho del cuerpo residual.
    n_blocks : int, default 4
        Cantidad de bloques residuales.
    dropout : float, default 0.15
        Dropout dentro de cada bloque residual.

    Shapes
    ------
    forward(voxels):
        voxels : (B, in_voxels)
        return : (B, out_dim)
    """

    def __init__(
        self,
        in_voxels: int,
        out_dim: int,
        hidden_dim: int = 4096,
        n_blocks: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        if in_voxels <= 0:
            raise ValueError(f"in_voxels debe ser > 0, recibido {in_voxels}")
        if out_dim <= 0:
            raise ValueError(f"out_dim debe ser > 0, recibido {out_dim}")
        if n_blocks < 1:
            raise ValueError(f"n_blocks debe ser >= 1, recibido {n_blocks}")

        self.in_voxels = in_voxels
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks

        self.input_proj = nn.Sequential(
            nn.Linear(in_voxels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList(
            _ResidualMLPBlock(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(n_blocks)
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        if voxels.dim() != 2:
            raise ValueError(
                f"voxels debe ser 2D (B, in_voxels), recibido shape={tuple(voxels.shape)}"
            )
        if voxels.size(-1) != self.in_voxels:
            raise ValueError(
                f"última dim debe ser {self.in_voxels}, recibido {voxels.size(-1)}"
            )

        h = self.input_proj(voxels)          # (B, hidden_dim)
        for block in self.blocks:
            h = block(h)                     # (B, hidden_dim)
        out = self.output_proj(h)            # (B, out_dim)
        return out


# =============================================================================
# Loss compuesta: InfoNCE bidireccional + MSE + Cosine
# =============================================================================

class MindEyeLoss(nn.Module):
    """
    Pérdida híbrida para alineación fMRI ↔ CLIP.

    Componentes
    -----------
    1. **InfoNCE bidireccional** (principal). Trata el batch como negativos
       in-batch: para cada `pred_i`, su positivo es `target_i` y los
       `target_{j≠i}` son negativos (y simétrico para target→pred). Es la
       razón por la que la red aprende geometría del manifold (no sólo
       reducir distancia L2).
    2. **MSE auxiliar**. Ancla a la magnitud del embedding objetivo. Sin
       ella la solución contrastiva podría escalar libremente.
    3. **Cosine auxiliar** (`1 - cos_sim`). Refuerza alineación direccional;
       redundante con InfoNCE pero da señal de gradiente densa al inicio
       del entrenamiento.

    Temperatura
    -----------
    Sigue el patrón CLIP/SigLIP: parametrizamos `logit_scale = log(1/τ)` como
    learnable y aplicamos `logit_scale.exp()` al producto interno. Init en
    `log(1/0.07) ≈ 2.6593`. Se clampa a ≤ `log(100)` antes del `exp` para
    evitar overflow en bf16 (donde `exp(>11.09)` ya satura `inf`).

    Pesos
    -----
    `w_nce=1.0`, `w_mse=0.3`, `w_cos=0.3`. Empíricamente equilibrados; el
    auxiliar pesa < principal para no dominar el gradiente.

    Forward
    -------
    pred_embeds          : (B, D)  salida del backbone
    target_clip_embeds   : (B, D)  embedding CLIP target
    return : dict con claves {"loss", "loss_nce", "loss_mse", "loss_cos"}.
    """

    # Tope de log-temperatura para evitar exp() inestable en bf16.
    LOGIT_SCALE_MAX = 4.6052  # log(100)

    def __init__(
        self,
        w_nce: float = 1.0,
        w_mse: float = 0.3,
        w_cos: float = 0.3,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        if init_temperature <= 0:
            raise ValueError(f"init_temperature debe ser > 0, recibido {init_temperature}")

        self.w_nce = w_nce
        self.w_mse = w_mse
        self.w_cos = w_cos

        # logit_scale = log(1/τ); exp(logit_scale) = 1/τ
        init_logit_scale = torch.log(torch.tensor(1.0 / init_temperature))
        self.logit_scale = nn.Parameter(init_logit_scale)

    def forward(
        self,
        pred_embeds: torch.Tensor,
        target_clip_embeds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if pred_embeds.shape != target_clip_embeds.shape:
            raise ValueError(
                "pred y target deben tener mismo shape; "
                f"pred={tuple(pred_embeds.shape)} target={tuple(target_clip_embeds.shape)}"
            )
        if pred_embeds.dim() != 2:
            raise ValueError(
                f"esperado (B, D), recibido shape={tuple(pred_embeds.shape)}"
            )

        batch_size = pred_embeds.size(0)
        device = pred_embeds.device

        # ---------- InfoNCE bidireccional ----------
        # Normalización L2 estable (F.normalize añade eps interno).
        pred_n = F.normalize(pred_embeds, dim=-1)
        targ_n = F.normalize(target_clip_embeds, dim=-1)

        # Clamp + exp en fp32 evita saturación en bf16.
        scale = self.logit_scale.clamp(max=self.LOGIT_SCALE_MAX).exp()

        # logits[i, j] = scale * <pred_i, targ_j>
        logits_p2t = scale * pred_n @ targ_n.t()      # (B, B)
        logits_t2p = logits_p2t.t()                   # (B, B)

        labels = torch.arange(batch_size, device=device)
        loss_p2t = F.cross_entropy(logits_p2t, labels)
        loss_t2p = F.cross_entropy(logits_t2p, labels)
        loss_nce = 0.5 * (loss_p2t + loss_t2p)

        # ---------- MSE auxiliar ----------
        loss_mse = F.mse_loss(pred_embeds, target_clip_embeds)

        # ---------- Cosine auxiliar (1 - cos) ----------
        cos_sim = (pred_n * targ_n).sum(dim=-1)       # (B,)
        loss_cos = (1.0 - cos_sim).mean()

        # ---------- Total ----------
        loss = (
            self.w_nce * loss_nce
            + self.w_mse * loss_mse
            + self.w_cos * loss_cos
        )

        return {
            "loss":     loss,
            "loss_nce": loss_nce.detach(),
            "loss_mse": loss_mse.detach(),
            "loss_cos": loss_cos.detach(),
        }


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    B, V, D = 4, 4000, 768

    backbone = MindEyeBackbone(in_voxels=V, out_dim=D)
    loss_fn = MindEyeLoss()

    voxels = torch.randn(B, V)
    target = torch.randn(B, D)

    pred = backbone(voxels)
    assert pred.shape == (B, D), f"shape mismatch: {pred.shape} != {(B, D)}"

    out = loss_fn(pred, target)
    expected_keys = {"loss", "loss_nce", "loss_mse", "loss_cos"}
    assert set(out.keys()) >= expected_keys, f"faltan claves: {expected_keys - set(out)}"
    assert out["loss"].dim() == 0, "loss debe ser escalar"
    assert torch.isfinite(out["loss"]), "loss no finita"

    out["loss"].backward()
    assert backbone.input_proj[0].weight.grad is not None, "no gradiente en input_proj"
    assert loss_fn.logit_scale.grad is not None, "no gradiente en logit_scale"

    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"[smoke] OK | params={n_params/1e6:.2f}M | pred={tuple(pred.shape)} "
          f"| loss={out['loss'].item():.4f} "
          f"(nce={out['loss_nce'].item():.4f} "
          f"mse={out['loss_mse'].item():.4f} "
          f"cos={out['loss_cos'].item():.4f})")
