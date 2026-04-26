#!/usr/bin/env bash
# =============================================================================
# run_remaining_subjects.sh — Despliegue Fase 2 sujetos restantes BOLD5000.
#
# Sujetos: CSI2, CSI3, CSI4 (CSI1 ya validado en run anterior).
#
# Pasos por sujeto:
#   3. train_adapter (Ridge fMRI -> CLIP)
#   4. visual_evaluator (SD 2.1 unCLIP, 113 test set, --steps 50)
#
# SKIPS (ya completados, no rehacer en remota):
#   - Paso 1: pip install -r requirements_py312.txt
#   - Paso 2: extract_vit_features (embeddings ViT-L/14 idénticos cross-subject,
#            ya escritos en {phase2_outputs}/clip_vit_l14/).
#
# Para forzar re-ejecución manual de los pasos saltados, usa run_production.sh.
#
# Outputs segregados por sujeto:
#   {phase2_outputs}/adapter/{CSI2,CSI3,CSI4}/{ridge_adapter.joblib,embeds_test.pt}
#   {eval_output}/{CSI2,CSI3,CSI4}/...
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Lista de sujetos a procesar. Editar aquí si se quiere reincluir CSI1.
SUBJECTS=(CSI2 CSI3 CSI4)

# ---------------------------------------------------------------------------
# Paso 1 (pip install) — DESACTIVADO. Dependencias ya instaladas.
# ---------------------------------------------------------------------------
# pip install -r requirements_py312.txt

# ---------------------------------------------------------------------------
# Paso 2 (extract_vit_features) — DESACTIVADO.
# Embeddings CLIP ViT-L/14 son función solo de los estímulos (idénticos para
# todos los sujetos BOLD5000), ya extraídos en el run de CSI1.
# ---------------------------------------------------------------------------
# STIM_DIR="$(python -c "import config; print(config.BOLD5000_CONFIG['stimuli_images'])")"
# python -m phase2.extract_vit_features --stimuli-dir "$STIM_DIR"

for SUBJ in "${SUBJECTS[@]}"; do
    echo
    echo "=============================================================================="
    echo "Sujeto $SUBJ — [3/4] train_adapter (Ridge fMRI -> CLIP)"
    echo "=============================================================================="
    python -m phase2.train_adapter --mode bold5000 --subject "$SUBJ"

    echo
    echo "=============================================================================="
    echo "Sujeto $SUBJ — [4/4] visual_evaluator (SD 2.1 unCLIP, 113 test, steps=50)"
    echo "=============================================================================="
    python -m phase2.visual_evaluator --subject "$SUBJ" --steps 50
done

echo
echo "Remaining subjects run completo: ${SUBJECTS[*]}"
