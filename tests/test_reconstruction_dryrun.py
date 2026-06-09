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


def test_embeddings_loader_renormalizes(mock_ecosystem):
    """load_adapter_embeddings debe entregar (N, 768) finito y con norma fija."""
    import torch
    from phase2.visual_evaluator import load_adapter_embeddings

    trial_ids, emb = load_adapter_embeddings(mock_ecosystem["embeds_test_pt"])
    assert emb.ndim == 2 and emb.shape[1] == 768
    assert len(trial_ids) == emb.shape[0]
    assert torch.isfinite(emb).all()
    # El fix de shrinkage renormaliza a norma ~12 por fila.
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 12.0), atol=1e-3)


def test_alignment_mismatch_raises(mock_ecosystem, monkeypatch):
    """Si las filas de embeds no cuadran con los stems del test, debe abortar."""
    import config
    from phase2 import visual_evaluator as ve

    monkeypatch.setitem(config.BOLD5000_CONFIG, "stim_lists_root", mock_ecosystem["stim_lists_root"])
    monkeypatch.setitem(config.BOLD5000_CONFIG, "repeated_list_txt", mock_ecosystem["repeated_list_txt"])
    monkeypatch.setitem(config.BOLD5000_CONFIG, "clip_targets_pt", mock_ecosystem["clip_targets_pt"])

    with pytest.raises(ValueError):
        ve.align_stems_to_embeddings("CSI1", n_rows=999)
