# Contrato de Interfaz: Migración MindEye (Fase 2)

Este documento define la interfaz estricta entre la arquitectura de modelos (`mindeye_models.py`) y el pipeline de entrenamiento (`train_mindeye.py`). **Ninguno de los dos módulos debe romper estas reglas.**

## 1. Dimensiones y Configuraciones Dinámicas
- **Dimensión Target (CLIP):** Ambos módulos deben leer dinámicamente la dimensión de salida desde la configuración global:
  ```python
  import config
  out_dim = config.SD_CONFIG["embedding_dim"]  # 768 para diffusers/stable-diffusion-2-1-unclip-i2i-l
  ```
- **Tamaño de Batch:** El fallback mínimo de seguridad para evitar OOM en la RTX 2070 (8GB) de dev es `batch_size=32`. Para la RTX 4070 Ti (12GB) de producción, se puede intentar `64`.

## 2. Exportaciones Requeridas (`phase2/mindeye_models.py`)

### `MindEyeBackbone(nn.Module)`
- **Constructor:** `__init__(self, in_voxels: int, out_dim: int, hidden_dim: int = 4096, n_blocks: int = 4, dropout: float = 0.15)`
- **Forward Signature:** `forward(self, voxels: torch.Tensor) -> torch.Tensor`
  - `voxels` shape: `(Batch, in_voxels)`
  - `return` shape: `(Batch, out_dim)`
- **Operaciones:** Debe ser compatible con *Automatic Mixed Precision* (AMP) bf16. Evitar operaciones inestables en precisión reducida.

### `MindEyeLoss(nn.Module)`
- **Constructor:** `__init__(self)`
- **Forward Signature:** `forward(self, pred_embeds: torch.Tensor, target_clip_embeds: torch.Tensor) -> dict[str, torch.Tensor]`
  - Recibe tensores de shape `(Batch, out_dim)`
  - **Retorno Obligatorio:** Un diccionario con al menos las siguientes claves:
    - `"loss"`: La pérdida total (escalar) que será usada para `loss.backward()`.
    - `"loss_nce"`: Pérdida contrastiva InfoNCE bidireccional.
    - `"loss_mse"`: MSE tradicional (para referencia/auxiliar).
    - `"loss_cos"`: Similitud Coseno (para referencia/auxiliar).

## 3. Requerimientos de Entrenamiento (`phase2/train_mindeye.py`)
- **Dependencias:** Debe importar `MindEyeBackbone` y `MindEyeLoss` desde `phase2.mindeye_models`.
- **Checkpointing:** Guardar el modelo como `best_mindeye_model.pt` basado en la mejor validación de *Pairwise Accuracy*, **no** en el loss más bajo.
- **Métricas:** Volcar `metrics_mindeye.json` por cada epoch para que puedan ser visualizadas/leídas sin recompilar.
- **Salida Final:** Al finalizar, generar `embeds_test.pt` con shape `(113, out_dim)` ordenado por `split.trial_ids_test`, manteniendo 100% compatibilidad con el actual `visual_evaluator.py`.
