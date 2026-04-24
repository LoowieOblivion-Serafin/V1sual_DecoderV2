# STATE - Fase 2: BOLD5000 + SD 2.1 unCLIP
**Fecha:** 23 de Abril de 2026

**Estado de la Arquitectura (Refactorización en curso):**
- **Pipeline:** BOLD5000_Stimuli → CLIP ViT-L/14 → Ridge Adapter → SD 2.1 unCLIP-L (frozen).
- **Módulos Funcionales (No tocar):** `sd_decoder.py`, `phase2_run_sd.py`, `phase2/adapter_ridge.py`, `phase2/extract_vit_features.py`, `phase2/train_adapter.py`.
- **Módulos a Refactorizar (NSD a BOLD5000):** `config.py` (nombres y paths), `phase2/nsd_loader.py` (derogar por BOLD5000), `phase2/mock_data.py` (renombrar a MockSplit), `evaluation.py` (actualizar docstrings).
- **Limpieza completada:** VQGAN, VGG19 y dependencias legacy han sido purgadas de la arquitectura.

**Decisiones Técnicas Arquitectónicas:**
1. **Extracción de Betas (Fast Route) [DESBLOQUEADO]:** El usuario ha descargado exitosamente los ROIs release. Los archivos `.mat` pre-calculados (ej. `CSI1_ROIs_TR34.mat`) se encuentran disponibles en `BOLD5000_ROIs/ROIs/CSI{N}/mat/`. La ruta lenta GLM LSS queda descartada definitivamente.
2. **Data Loader (`bold5000_loader.py`):** Implementado y aprobado. Maneja lectura secuencial de estímulos y previene leakage mediante z-score calculado solo en train.
3. **Gestión de Entornos (Local vs RTX 4070 Ti):** `config.py` refactorizado exitosamente. Utiliza resolución dinámica vía `pathlib` y soporta variables de entorno `ACECOM_*` para anulación en la RTX 4070 Ti. El proyecto es oficialmente portátil.
4. **Integración Final (Completada):** El legacy `nsd_loader` fue purgado. `train_adapter.py` ahora consume nativamente la data de BOLD5000 configurada.
5. **Evaluador Visual [VALIDADO - LISTO PARA PRODUCCIÓN]:** `phase2/visual_evaluator.py` ejercita el pipeline end-to-end (embeds adapter → SD 2.1 unCLIP → GT rglob → pair PNG + grid collage) con resolución dinámica 100% vía `config.py` + `ACECOM_*`. Smoke-test sintético en CPU confirmado el 2026-04-23: tensores `(N, 768)` encajan con el contrato del pipeline, `get_ordered_test_stems` alinea 1:1 con las filas de `embeds_test.pt`, IO de collages correcto. Pendiente despliegue en RTX 4070 Ti para inferencia real (primer run descarga ~5GB de pesos SD).
   - Helper operativo: `phase2/create_mock_assets.py` genera ecosistema BOLD5000 sintético bajo `--root` para re-validar en cualquier entorno sin GPU.
   - Flag `--dry-run` en el evaluador: skip SD load, substituye stub PIL. Útil para CI y smoke de estructura.