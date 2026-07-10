@echo off
REM ============================================================
REM  compile.bat - Reconstruye main.pdf desde cero
REM  Uso: doble click, o desde CMD:  compile.bat
REM  Requisitos: MiKTeX (pdflatex + bibtex en PATH)
REM ============================================================

setlocal EnableDelayedExpansion
pushd "%~dp0"

echo.
echo ============================================================
echo  TESIS - PIPELINE LATEX
echo  Directorio: %CD%
echo ============================================================
echo.

REM --- Verificar pdflatex disponible ---
where pdflatex >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pdflatex no encontrado en PATH.
    echo Instala MiKTeX o agrega su bin al PATH.
    pause
    exit /b 1
)

REM --- Limpieza de artefactos previos ---
echo [1/6] Limpiando artefactos intermedios...
del /Q main.aux main.bbl main.blg main.log main.toc main.lof main.lot main.out main.equ main.lol main.pdf texput.log 2>nul
del /Q Chapters\*.aux 2>nul
del /Q Appendix\*.aux 2>nul

REM --- Paso 1: pdflatex (genera .aux con \cite pendientes) ---
echo [2/6] pdflatex pasada 1...
pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul
if errorlevel 1 (
    echo [ERROR] Fallo pdflatex pasada 1. Revisa main.log.
    pause
    exit /b 1
)

REM --- Paso 2: bibtex (resuelve referencias) ---
echo [3/6] bibtex...
bibtex main
if errorlevel 1 (
    echo [WARN] bibtex devolvio codigo de error. Revisa main.blg.
)

REM --- Paso 3: pdflatex (incorpora .bbl) ---
echo [4/6] pdflatex pasada 2...
pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul
if errorlevel 1 (
    echo [ERROR] Fallo pdflatex pasada 2.
    pause
    exit /b 1
)

REM --- Paso 4: pdflatex (resuelve cross-refs finales) ---
echo [5/6] pdflatex pasada 3...
pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul
if errorlevel 1 (
    echo [ERROR] Fallo pdflatex pasada 3.
    pause
    exit /b 1
)

REM --- Diagnostico final ---
echo [6/6] Verificando salida...
if not exist main.pdf (
    echo [ERROR] main.pdf no se genero.
    pause
    exit /b 1
)

REM --- Reporte de warnings clave ---
echo.
echo ============================================================
echo  REPORTE
echo ============================================================
for %%F in (main.pdf) do echo  main.pdf: %%~zF bytes
findstr /C:"Citation" /C:"undefined" main.log | findstr /I "undefined" >nul
if not errorlevel 1 (
    echo  [WARN] Hay citas no resueltas:
    findstr /C:"undefined" main.log | findstr /I "Citation"
) else (
    echo  Citas: OK
)
findstr /C:"Reference" main.log | findstr /I "undefined" >nul
if not errorlevel 1 (
    echo  [WARN] Hay referencias cruzadas no resueltas:
    findstr /C:"undefined" main.log | findstr /I "Reference"
) else (
    echo  Cross-refs: OK
)
echo ============================================================
echo  DONE - main.pdf listo
echo ============================================================
echo.

REM --- Abrir PDF (opcional, comenta si no quieres) ---
start "" main.pdf

popd
endlocal
