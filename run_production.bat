@echo off
REM =============================================================================
REM run_production.bat - Despliegue completo Fase 2 en RTX 4070 Ti (Windows).
REM
REM Secuencia:
REM   1. Instala dependencias Python (diffusers, transformers, accelerate...).
REM   2. Extrae embeddings CLIP ViT-L/14 de los 4916 estimulos BOLD5000.
REM   3. Entrena adapter Ridge fMRI -> CLIP para CSI1, dumpea embeds_test.pt.
REM   4. Corre visual_evaluator sobre las 113 imagenes del test set (sin
REM      --limit) forzando --steps 50 para maxima fidelidad en el run
REM      automatico (config.SD_CONFIG mantiene 25 como default rapido).
REM
REM Paths resueltos via config.py + variables ACECOM_*. Override antes de correr:
REM
REM   set ACECOM_BOLD5000_ROIS_ROOT=D:\data\BOLD5000_ROIs
REM   set ACECOM_BOLD5000_STIMULI_ROOT=D:\data\BOLD5000_Stimuli
REM   set ACECOM_PHASE2_OUTPUTS=D:\scratch\phase2_outputs
REM   set ACECOM_EVAL_OUTPUT=D:\scratch\output_reconstructions_sd21
REM   set ACECOM_HF_CACHE=D:\models\hf
REM =============================================================================

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

echo ==============================================================================
echo [1/4] pip install -r requirements_py312.txt
echo ==============================================================================
python -m pip install -r requirements_py312.txt || exit /b 1

echo.
echo ==============================================================================
echo [2/4] extract_vit_features (CLIP ViT-L/14 sobre BOLD5000 stimuli)
echo ==============================================================================
for /f "usebackq delims=" %%i in (`python -c "import config; print(config.BOLD5000_CONFIG['stimuli_images'])"`) do set STIM_DIR=%%i
echo stimuli-dir = %STIM_DIR%
python -m phase2.extract_vit_features --stimuli-dir "%STIM_DIR%" || exit /b 1

echo.
echo ==============================================================================
echo [3/4] train_adapter (Ridge fMRI -> CLIP, sujeto CSI1)
echo ==============================================================================
python -m phase2.train_adapter --mode bold5000 --subject CSI1 || exit /b 1

echo.
echo ==============================================================================
echo [4/4] visual_evaluator (SD 2.1 unCLIP sobre 113 test set)
echo ==============================================================================
python -m phase2.visual_evaluator --subject CSI1 --steps 50 || exit /b 1

echo.
echo Production run completo.
endlocal
