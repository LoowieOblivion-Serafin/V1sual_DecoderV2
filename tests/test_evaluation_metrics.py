"""
Tests de `evaluation.py`: métricas y, sobre todo, el WIRING de datos que estaba
roto (config.NSD_CONFIG inexistente + sufijo `_sd_unclip.png` que no casaba con
`{stem}_recon.png`). Verifica que el emparejamiento recon↔GT por stem funciona
con el layout real que escribe `visual_evaluator.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from PIL import Image  # noqa: E402  (PIL siempre presente en este repo)


# ---------------------------------------------------------------------------
# Métricas de píxel
# ---------------------------------------------------------------------------

def test_pixel_metrics_identical_is_perfect():
    pytest.importorskip("skimage")
    pytest.importorskip("scipy")
    from evaluation import pixel_metrics

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    m = pixel_metrics(img, img)
    assert m["ssim"] == pytest.approx(1.0, abs=1e-6)
    assert m["pixcorr"] == pytest.approx(1.0, abs=1e-6)


def test_pixel_metrics_independent_low_corr():
    pytest.importorskip("scipy")
    from evaluation import pixel_metrics

    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    m = pixel_metrics(a, b)
    assert abs(m["pixcorr"]) < 0.2


# ---------------------------------------------------------------------------
# Wiring de datos (el bug que arreglamos)
# ---------------------------------------------------------------------------

def _write_img(path, color, size=32):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size, size, 3), color, dtype=np.uint8)).save(path)


def test_build_gt_index_and_load_recons_match_by_stem(tmp_path):
    from evaluation import align_pairs, build_gt_index, load_reconstructions

    # Árbol de estímulos anidado (como BOLD5000) y recon en el layout real.
    stimuli = tmp_path / "stimuli"
    _write_img(stimuli / "Scene" / "alpha.jpg", 10)
    _write_img(stimuli / "COCO" / "beta.jpg", 250)

    recon_root = tmp_path / "recons"
    _write_img(recon_root / "CSI1" / "reconstructions" / "alpha_recon.png", 11)
    _write_img(recon_root / "CSI1" / "reconstructions" / "beta_recon.png", 240)

    gt_index = build_gt_index(stimuli)
    assert set(gt_index) == {"alpha", "beta"}

    recons = load_reconstructions("CSI1", recon_root)
    assert set(recons) == {"alpha", "beta"}

    pairs = align_pairs("CSI1", gt_index, recons)
    assert len(pairs) == 2
    assert {p.label for p in pairs} == {"alpha", "beta"}


def test_load_recons_supports_flat_legacy_layout(tmp_path):
    from evaluation import load_reconstructions

    recon_root = tmp_path / "recons"
    _write_img(recon_root / "CSI2" / "gamma_recon.png", 100)  # sin subdir reconstructions/
    recons = load_reconstructions("CSI2", recon_root)
    assert set(recons) == {"gamma"}


def test_parse_args_uses_bold5000_not_nsd():
    """Regresión: antes apuntaba a config.NSD_CONFIG (inexistente) y crasheaba."""
    import sys

    import config
    from evaluation import parse_args

    argv = sys.argv
    sys.argv = ["evaluation", "--subjects", "CSI1"]
    try:
        args = parse_args()
    finally:
        sys.argv = argv
    assert args.subjects == ["CSI1"]
    assert set(config.BOLD5000_SUBJECTS) == {"CSI1", "CSI2", "CSI3", "CSI4"}
