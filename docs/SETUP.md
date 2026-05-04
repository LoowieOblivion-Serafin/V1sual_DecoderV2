# Guía de Instalación — Fase 2 (NSD + SD 2.1 unCLIP)

Instrucciones para preparar el entorno de inferencia con Stable Diffusion 2.1 unCLIP sobre Natural Scenes Dataset.

---

## 1. Requisitos

- **GPU**: NVIDIA con ≥ 8 GB VRAM (RTX 3070 mínimo; RTX 4070 Ti / 12 GB recomendado).
- **Python**: 3.12.
- **CUDA**: 12.1 o superior.
- **Disco**: ≥ 15 GB libres en SSD (pesos SD 2.1 unCLIP ~5 GB + NSD subset).

---

## 2. Entorno virtual (Windows)

```powershell
python -m venv env_tesis
.\env_tesis\Scripts\Activate.ps1
```

---

## 3. PyTorch + CUDA

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verificación:

```powershell
python -c "import torch; print('CUDA Ready:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

---

## 4. Dependencias del proyecto

```powershell
pip install -r requirements_py312.txt
```

Incluye `diffusers`, `transformers`, `xformers`, `accelerate`, `nibabel`, `nilearn`, `h5py`, `scikit-learn`, `lpips`, etc.

> **xformers**: inyecta atención memory-efficient en SD; reduce VRAM pico de ~9 GB a ~5-6 GB en bf16.

---

## 5. Estructura de datos esperada

```text
.
├── BOLD5000_ROIs/                # Matrices .mat de extracción de vóxeles (CSI1-4)
├── BOLD5000_Stimuli/             # Imágenes mostradas y orden de presentación
├── models_hf/                    # (auto) Cache de pesos HF
└── phase2_outputs/               # (auto) Salidas del entrenamiento y extracción
```

**Atención:** Los datos de BOLD5000 se deben descargar manualmente (ver enlaces en el `README.md`) y extraerse en la raíz del repositorio. No intentes subirlos a GitHub ya que superan el límite de 5GB.

---

## 6. Smoke tests

Smoke test del pipeline (sin datos NSD, usa embed dummy aleatorio):

```powershell
python sd_decoder.py
```

Inferencia desde embeddings reales del adapter BOLD5000 (post-entrenamiento):

```powershell
python phase2_run_sd.py --subject CSI1 --embeds phase2_outputs/adapter/CSI1/embeds_test.pt --limit 3
```

Una corrida exitosa descarga los pesos `safetensors` desde HF Hub a `models_hf/`, instancia `DPMSolverMultistepScheduler` (25 pasos), e imprime:

```
[CSI1] (1/3) <stem> -> CSI1_<stem>_recon.png
```

---

## 7. Problemas frecuentes

1. **OOM en RTX 3070 (8 GB)**: confirmar que xformers se cargó correctamente y considerar `enable_attention_slicing` en `sd_decoder.py`.
2. **CUDA no disponible**: reinstalar PyTorch con el index correcto (`cu121`) y verificar drivers NVIDIA actualizados.
3. **`diffusers` versión antigua**: requiere `>= 0.27.0` para `StableUnCLIPImg2ImgPipeline` con CFG negativa.
