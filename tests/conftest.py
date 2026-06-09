"""
Configuración compartida de pytest para la suite de la Fase 2.

- Inserta `src/` en `sys.path` (mismo contrato que runtime).
- Expone el fixture `mock_ecosystem`: genera un mini-BOLD5000 sintético en
  un directorio temporal vía `phase2.create_mock_assets`, sin tocar el dataset
  real ni requerir GPU/diffusers.

Los tests pesados usan `pytest.importorskip(...)` para degradar con elegancia
si falta torch / matplotlib / skimage en el entorno de CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# --- src en el path (idéntico a lo que hace cada módulo en runtime) ---
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture
def mock_ecosystem(tmp_path):
    """
    Crea un ecosistema BOLD5000 sintético completo en `tmp_path/mock`:
        stimuli (Presented_Stimuli + Presentation_Lists + repeated list),
        clip_targets.pt y adapter/embeds_test.pt.

    Devuelve el dict de paths de `create_mock_ecosystem` + 'root'.
    """
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    from phase2.create_mock_assets import create_mock_ecosystem

    root = tmp_path / "mock"
    paths = create_mock_ecosystem(root, n=6, subject="CSI1", seed=123)
    paths["root"] = root
    paths["stim_lists_root"] = paths["stimuli_root"] / "Stimuli_Presentation_Lists"
    return paths
