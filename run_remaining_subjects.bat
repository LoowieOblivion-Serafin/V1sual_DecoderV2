@echo off
REM =============================================================================
REM run_remaining_subjects.bat — Despliegue Fase 2 sujetos restantes BOLD5000.
REM
REM Sujetos: CSI2, CSI3, CSI4 (CSI1 ya validado en run anterior).
REM
REM Pasos por sujeto:
REM   3. train_adapter (Ridge fMRI -> CLIP)
REM   4. visual_evaluator (SD 2.1 unCLIP, 113 test set, --steps 50)
REM
REM SKIPS (ya completados, no rehacer en remota):
REM   - Paso 1: pip install -r requirements_py312.txt
REM   - Paso 2: extract_vit_features (embeddings ViT-L/14 idénticos cross-subject,
REM            ya escritos en {phase2_outputs}/clip_vit_l14/).
REM
REM Para forzar re-ejecución manual de los pasos saltados, usa run_production.bat.
REM
REM Outputs segregados por sujeto:
REM   {phase2_outputs}/adapter/{CSI2,CSI3,CSI4}/{ridge_adapter.joblib,embeds_test.pt}
REM   {eval_output}/{CSI2,CSI3,CSI4}/...
REM =============================================================================

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

REM Lista de sujetos a procesar. Editar aquí si se quiere reincluir CSI1.
set SUBJECTS=CSI2 CSI3 CSI4

REM ---------------------------------------------------------------------------
REM Paso 1 (pip install) — DESACTIVADO. Dependencias ya instaladas.
REM ---------------------------------------------------------------------------
REM echo [1/4] pip install -r requirements_py312.txt
REM python -m pip install -r requirements_py312.txt || exit /b 1

REM ---------------------------------------------------------------------------
REM Paso 2 (extract_vit_features) — DESACTIVADO.
REM Embeddings CLIP ViT-L/14 son función solo de los estímulos (idénticos para
REM todos los sujetos BOLD5000), ya extraídos en el run de CSI1.
REM ---------------------------------------------------------------------------
REM for /f "usebackq delims=" %%i in (`python -c "import config; print(config.BOLD5000_CONFIG['stimuli_images'])"`) do set STIM_DIR=%%i
REM python -m phase2.extract_vit_features --stimuli-dir "%STIM_DIR%" || exit /b 1

for %%S in (%SUBJECTS%) do (
    echo.
    echo ==============================================================================
    echo Sujeto %%S — [3/4] train_adapter Ridge fMRI -^> CLIP
    echo ==============================================================================
    python -m phase2.train_adapter --mode bold5000 --subject %%S || exit /b 1

    echo.
    echo ==============================================================================
    echo Sujeto %%S — [4/4] visual_evaluator SD 2.1 unCLIP, 113 test, steps=50
    echo ==============================================================================
    python -m phase2.visual_evaluator --subject %%S --steps 50 || exit /b 1
)

echo.
echo Remaining subjects run completo: %SUBJECTS%
endlocal
