#!/usr/bin/env bash
# =============================================================================
# run_production.sh — Despliegue completo Fase 2 en RTX 4070 Ti (Linux/WSL).
#
# Secuencia:
#   1. Instala dependencias Python (diffusers, transformers, accelerate, etc.).
#   2. Extrae embeddings CLIP ViT-L/14 de los 4916 estímulos BOLD5000.
#   3. Entrena adapter Ridge fMRI -> CLIP para CSI1 y dumpea embeds_test.pt.
#   4. Corre visual_evaluator sobre las 113 imagenes del test set (sin --limit)
#      con steps default de config.SD_CONFIG (25).
#
# Paths resueltos via config.py + variables de entorno ACECOM_*. Exportar
# overrides antes de correr si los datos no viven bajo el repo root:
#
#   export ACECOM_BOLD5000_ROIS_ROOT=/mnt/data/BOLD5000_ROIs
#   export ACECOM_BOLD5000_STIMULI_ROOT=/mnt/data/BOLD5000_Stimuli
#   export ACECOM_PHASE2_OUTPUTS=/mnt/scratch/phase2_outputs
#   export ACECOM_EVAL_OUTPUT=/mnt/scratch/output_reconstructions_sd21
#   export ACECOM_HF_CACHE=/mnt/models/hf
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "[1/4] pip install -r requirements_py312.txt"
echo "=============================================================================="
pip install -r requirements_py312.txt

echo
echo "=============================================================================="
echo "[2/4] extract_vit_features (CLIP ViT-L/14 sobre BOLD5000 stimuli)"
echo "=============================================================================="
STIM_DIR="$(python -c "import config; print(config.BOLD5000_CONFIG['stimuli_images'])")"
echo "stimuli-dir = $STIM_DIR"
python -m phase2.extract_vit_features --stimuli-dir "$STIM_DIR"

echo
echo "=============================================================================="
echo "[3/4] train_adapter (Ridge fMRI -> CLIP, sujeto CSI1)"
echo "=============================================================================="
python -m phase2.train_adapter --mode bold5000 --subject CSI1

echo
echo "=============================================================================="
echo "[4/4] visual_evaluator (SD 2.1 unCLIP sobre 113 test set)"
echo "=============================================================================="
python -m phase2.visual_evaluator --subject CSI1

echo
echo "Production run completo."
