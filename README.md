# Reconstrucción de Imágenes Mentales desde fMRI (Stable Diffusion 2.1 unCLIP + BOLD5000)

Proyecto de tesis (UNI · ACECOM) que decodifica imágenes desde actividad cerebral fMRI usando un adapter aprendido fMRI→CLIP-ViT-L/14 y el pipeline Stable Diffusion 2.1 unCLIP.

## Arquitectura

```
fMRI (BOLD5000 ROIs, ~10k vóxeles)
   │
   ▼
Adapter Ridge / MLP+LoRA  ──► z_CLIP ∈ R^768
   │
   ▼
SD 2.1 unCLIP UNet (frozen) + VAE decoder (frozen)
   │
   ▼
Imagen reconstruida
```

Componentes:
- **Dataset**: BOLD5000 (OpenNeuro) — fMRI, ROIs release (CSI1-4), estímulos COCO/ImageNet/Scene.
- **Encoder semántico**: `openai/clip-vit-large-patch14` (target del adapter).
- **Generador**: `diffusers/stable-diffusion-2-1-unclip-i2i-l` en bf16 + xformers.
- **Adapter**: ridge regression sklearn (baseline) o MLP+LoRA (avanzado).

## Estado del proyecto

Rama actual: `main`. Stack VQGAN+VGG19 (Fase 1) purgado. Pipeline BOLD5000 + SD 2.1 unCLIP integrado.

**Aviso de Datos (Límite GitHub 5GB):**
Debido a las restricciones de tamaño de GitHub, los datos de fMRI y los estímulos no se suben al repositorio. Debes descargarlos manualmente y extraerlos en la raíz del proyecto:
- **BOLD5000 ROIs (.mat)**: [Descargar desde Figshare](https://figshare.com/articles/dataset/BOLD5000/6459449?file=12965447). Extraer en `BOLD5000_ROIs/`.
- **BOLD5000 Stimuli**: [Descargar desde Dropbox](https://www.dropbox.com/s/5ie18t4rjjvsl47/BOLD5000_Stimuli.zip?dl=1). Extraer en `BOLD5000_Stimuli/`.

*(Nota: el archivo `bold5000_reordered_data.npy` es redundante para nuestra pipeline y puede ser descartado para ahorrar espacio).*

## Requisitos

- Python 3.12
- NVIDIA GPU ≥ 8 GB VRAM (RTX 3070 mínimo, RTX 4070 Ti recomendada para entrenamiento masivo)
- CUDA 12.1
- ~15 GB libres (pesos SD 2.1 unCLIP + cache HF)

## Estructura del proyecto

```text
.
├── BOLD5000_ROIs/                # (Ignorado en git) Matrices .mat de extracción de vóxeles
├── BOLD5000_Stimuli/             # (Ignorado en git) Imágenes mostradas y listas de presentación
├── models_hf/                    # (auto) Cache de pesos diffusers / transformers
├── output_sd_reconstructions/    # (auto) PNGs reconstruidos por sujeto
├── phase2_outputs/               # (auto) Salidas del modelo adapter entrenado y features CLIP
├── phase2/                       # Módulos del pipeline de extracción y entrenamiento
├── Papers/                       # Bibliografía y referencias
├── sd_decoder.py                 # Pipeline core SD 2.1 unCLIP
├── phase2_run_sd.py              # Orquestador de inferencia desde embeddings
├── evaluation.py                 # Métricas: SSIM, PixCorr, PSNR, LPIPS, CLIP, pairwise-ID
├── config.py                     # Paths unificados dinámicos (Local/RTX4070Ti)
├── requirements_py312.txt        # Dependencias
├── MIGRATION.md                  # Racional del pivote a BOLD5000 + SD 2.1 unCLIP
├── Hoja_de_Ruta_Tesis.md         # Roadmap de tesis
└── SETUP.md                      # Instalación
```

## Instalación

Ver `SETUP.md`.

## Ejecución

Una vez entrenado el adapter y disponibles los embeddings 768-d:

```powershell
# Inferencia sobre un sujeto (CSI1):
python phase2_run_sd.py --subject CSI1 --embeds phase2_outputs/adapter/CSI1/embeds_test.pt

# Smoke test (5 trials):
python phase2_run_sd.py --subject CSI1 --embeds phase2_outputs/adapter/CSI1/embeds_test.pt --limit 5
```

Evaluación:

```powershell
python evaluation.py --subjects CSI1 --recon-dir output_sd_reconstructions --gt-dir BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli
```

## Créditos

Tesis de pregrado UNI (2025-2026), Alvaro Taipe Cotrina · Grupo ACECOM.
Base teórica: Koide-Majima et al. (2024); SOTA de referencia: MindEye2 (Scotti et al., ICML 2024), Brain-Diffuser (Ozcelik 2023), Takagi-Nishimoto 2023.
