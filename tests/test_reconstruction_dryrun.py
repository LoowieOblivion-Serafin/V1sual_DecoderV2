"""
Test E2E (mock) del pipeline de RECONSTRUCCIÓN — el headline de esta suite.

La reconstrucción "de verdad" corre en la Máquina B tras un `git pull` (carga
SD 2.1 unCLIP, ~5GB de pesos, GPU). Aquí validamos TODA la estructura del
pipeline en CPU y sin diffusers usando `--dry-run`:

    embeds_test.pt  → load + renorm
        → alineación stems↔filas
        → lookup de Ground Truth (rglob COCO/ImageNet/Scene)
        → render de pares GT|Recon + grid agregado
        → resumen con contadores ok/missing/failed

Si este test pasa, el único componente sin ejercitar es el forward de SD 2.1
unCLIP (sustituido por un stub PIL determinista). Cualquier regresión de IO,
shapes o alineación se detecta ANTES de gastar GPU en la Máquina B.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("PIL")


def test_reconstruction_pipeline_dryrun(mock_ecosystem, tmp_path, monkeypatch):
    import config
    from phase2 import visual_evaluator as ve

    # Apunta los lookups internos (get_ordered_test_stems) al ecosistema mock.
    monkeypatch.setitem(config.BOLD5000_CONFIG, "stim_lists_root", mock_ecosystem["stim_lists_root"])
    monkeypatch.setitem(config.BOLD5000_CONFIG, "repeated_list_txt", mock_ecosystem["repeated_list_txt"])
    monkeypatch.setitem(config.BOLD5000_CONFIG, "clip_targets_pt", mock_ecosystem["clip_targets_pt"])

    out_base = tmp_path / "eval_out"
    summary = ve.run_evaluation(
        subject="CSI1",
        embeds_path=mock_ecosystem["embeds_test_pt"],
        stimuli_root=mock_ecosystem["stimuli_root"],
        out_base=out_base,
        num_inference_steps=2,
        guidance_scale=1.0,
        noise_level=0,
        seed=0,
        limit=None,
        empty_cache_every=0,
        dpi=60,
        grid_rows=4,
        use_cpu=True,
        dry_run=True,
    )

    # --- Contadores ---
    assert summary["failed"] == 0, summary
    assert summary["ok"] >= 1, summary
    assert summary["subject"] == "CSI1"

    # --- Artefactos en disco ---
    subj = out_base / "CSI1"
    recons = list((subj / "reconstructions").glob("*_recon.png"))
    pairs = list((subj / "pairs").glob("*_compare.png"))
    assert recons, "no se generó ninguna reconstrucción PNG"
    assert pairs, "no se generó ningún par GT|Recon"
    assert (subj / "CSI1_grid.png").exists(), "falta el grid agregado"


def test_embeddings_loader_norm_modes(mock_ecosystem):
    """norm_mode 'ridge'/'unit'/'none' producen las normas esperadas; 'none' no toca."""
    import torch
    from phase2.visual_evaluator import load_adapter_embeddings

    path = mock_ecosystem["embeds_test_pt"]

    _, emb_ridge = load_adapter_embeddings(path, norm_mode="ridge", norm_scale=12.0)
    assert emb_ridge.ndim == 2 and emb_ridge.shape[1] == 768
    assert torch.isfinite(emb_ridge).all()
    assert torch.allclose(emb_ridge.norm(dim=-1), torch.full((emb_ridge.shape[0],), 12.0), atol=1e-3)

    _, emb_unit = load_adapter_embeddings(path, norm_mode="unit")
    assert torch.allclose(emb_unit.norm(dim=-1), torch.ones(emb_unit.shape[0]), atol=1e-4)

    # 'none' (nuevo default): la magnitud cruda se preserva tal cual del .pt.
    raw = torch.load(path, map_location="cpu")["embeddings"].float()
    _, emb_none = load_adapter_embeddings(path, norm_mode="none")
    assert torch.allclose(emb_none, raw, atol=1e-5)

    with pytest.raises(ValueError):
        load_adapter_embeddings(path, norm_mode="bogus")


def test_alignment_mismatch_raises(mock_ecosystem, monkeypatch):
    """Si las filas de embeds no cuadran con los stems del test, debe abortar."""
    import config
    from phase2 import visual_evaluator as ve

    monkeypatch.setitem(config.BOLD5000_CONFIG, "stim_lists_root", mock_ecosystem["stim_lists_root"])
    monkeypatch.setitem(config.BOLD5000_CONFIG, "repeated_list_txt", mock_ecosystem["repeated_list_txt"])
    monkeypatch.setitem(config.BOLD5000_CONFIG, "clip_targets_pt", mock_ecosystem["clip_targets_pt"])

    with pytest.raises(ValueError):
        ve.align_stems_to_embeddings("CSI1", n_rows=999)
