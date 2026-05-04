"""
Fase 2 — adapter fMRI→CLIP-ViT-L/14 sobre BOLD5000 (pivote OpenNeuro).

Módulos:
    loader             — facade: despacha entre 'bold5000' (real) y 'mock' (sintético).
    bold5000_loader    — loader real: ROIs .mat + presentation lists + targets CLIP.
    mock_data          — generador sintético para smoke tests sin disco.
    adapter_ridge      — baseline Ridge (sklearn) fMRI → 768-d CLIP.
    extract_vit_features — extrae embeddings CLIP ViT-L/14 sobre stimuli BOLD5000.
    train_adapter      — script end-to-end: entrena adapter + dump embeddings_test.
"""
