@echo off
REM =============================================================================
REM run_all_evaluations_test2.bat — Generar todas las reconstrucciones juntas
REM =============================================================================
REM Este script ejecuta ÚNICAMENTE el pipeline de evaluación visual (SD 2.1)
REM para todos los sujetos usando los adaptadores Ridge ya entrenados.
REM Las imágenes se guardarán en una nueva carpeta independiente.

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

REM 1. Configurar la carpeta de destino a 'output_reconstruccions_test2'
REM Asumiendo que D:\scratch es tu base, o usando ruta relativa.
REM Usamos el mismo patrón que el archivo batch anterior.
set ACECOM_EVAL_OUTPUT=output_reconstruccions_test2

set SUBJECTS=CSI1 CSI2 CSI3 CSI4

for %%S in (%SUBJECTS%) do (
    echo.
    echo ==============================================================================
    echo Sujeto %%S — [4/4] visual_evaluator SD 2.1 unCLIP, 113 test, steps=50
    echo ==============================================================================
    python -m phase2.visual_evaluator --subject %%S --steps 50 || exit /b 1
)

echo.
echo ==============================================================================
echo Evaluacion completa para todos los sujetos en %ACECOM_EVAL_OUTPUT%
echo ==============================================================================
endlocal
