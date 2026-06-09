"""
Tests del comparador cross-subject (`phase2.compare_subjects`).

Verifica: intersección de stems entre sujetos, render de la figura
[GT | CSI1..CSI4] y el manejo de celdas faltantes (sujeto sin esa recon).
Sin torch/diffusers: sólo matplotlib + PIL.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")
from PIL import Image  # noqa: E402

from phase2 import compare_subjects as cs


def _img(path, color, size=32):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size, size, 3), color, dtype=np.uint8)).save(path)


def _build_eval_dir(tmp_path, recon_map):
    """recon_map: {subject: [stems]} → escribe {eval}/{subj}/reconstructions/{stem}_recon.png."""
    eval_dir = tmp_path / "eval"
    for subj, stems in recon_map.items():
        for stem in stems:
            _img(eval_dir / subj / "reconstructions" / f"{stem}_recon.png", 120)
    return eval_dir


def test_common_stems_intersection(tmp_path):
    eval_dir = _build_eval_dir(tmp_path, {
        "CSI1": ["a", "b", "c"],
        "CSI2": ["a", "b"],
        "CSI3": ["a", "b", "z"],
        "CSI4": ["a", "b"],
    })
    stems = cs.common_stems(eval_dir, ["CSI1", "CSI2", "CSI3", "CSI4"])
    assert stems == ["a", "b"]


def test_render_comparison_creates_figure(tmp_path):
    eval_dir = _build_eval_dir(tmp_path, {s: ["scene1", "scene2"] for s in ("CSI1", "CSI2", "CSI3", "CSI4")})
    stimuli = tmp_path / "stimuli"
    _img(stimuli / "Scene" / "scene1.jpg", 30)
    _img(stimuli / "Scene" / "scene2.jpg", 60)

    out = tmp_path / "cmp.png"
    summary = cs.render_comparison(
        stems=["scene1", "scene2"],
        eval_dir=eval_dir,
        stimuli_root=stimuli,
        subjects=["CSI1", "CSI2", "CSI3", "CSI4"],
        out_path=out,
        cell_px=48,
        dpi=60,
    )
    assert out.exists()
    assert summary["n_stems"] == 2
    assert summary["n_subjects"] == 4
    assert summary["missing_cells"] == 0


def test_render_marks_missing_subject_cells(tmp_path):
    # CSI4 NO reconstruyó 'scene1' → debe contarse como celda faltante (placeholder).
    eval_dir = _build_eval_dir(tmp_path, {
        "CSI1": ["scene1"], "CSI2": ["scene1"], "CSI3": ["scene1"], "CSI4": [],
    })
    stimuli = tmp_path / "stimuli"
    _img(stimuli / "Scene" / "scene1.jpg", 30)

    out = tmp_path / "cmp_missing.png"
    summary = cs.render_comparison(
        stems=["scene1"],
        eval_dir=eval_dir,
        stimuli_root=stimuli,
        subjects=["CSI1", "CSI2", "CSI3", "CSI4"],
        out_path=out,
        cell_px=48,
        dpi=60,
    )
    assert out.exists()
    assert summary["missing_cells"] == 1


def test_select_stems_explicit_overrides_limit(tmp_path):
    eval_dir = _build_eval_dir(tmp_path, {s: ["x", "y"] for s in ("CSI1", "CSI2", "CSI3", "CSI4")})
    stems = cs.select_stems(eval_dir, ["CSI1", "CSI2"], explicit=["y"], limit=None, shuffle=False, seed=0)
    assert stems == ["y"]
