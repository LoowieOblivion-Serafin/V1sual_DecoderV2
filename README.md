# Reconstrucción de Imágenes Mentales desde fMRI (Stable Diffusion 2.1 unCLIP + BOLD5000)

Proyecto de tesis (UNI · ACECOM) que decodifica imágenes desde actividad cerebral fMRI usando una arquitectura de aprendizaje profundo no lineal inspirada en **MindEye (MedARC-AI)** y el pipeline Stable Diffusion 2.1 unCLIP.

## Arquitectura (Nueva Fase: MindEye)

```text
fMRI (BOLD5000 ROIs, ~10k vóxeles)
   │
   ▼
[MindEyeBackbone] MLP Residual (n_blocks=4)  ──► z_CLIP ∈ R^768
(Entrenado con InfoNCE Contrastive Loss)
   │
   ▼
SD 2.1 unCLIP UNet (frozen) + VAE decoder (frozen)
   │
   ▼
Imagen reconstruida
```

**Motivación del cambio:** El baseline anterior basado en regresión lineal (Ridge) colapsaba la variabilidad semántica en el espacio CLIP, generando texto alucinado y formas genéricas. Hemos migrado a una red estocástica y profunda con pérdida probabilística InfoNCE (Contrastive Learning) para obligar a los vectores fMRI a aprender la métrica real del espacio latente visual.

Componentes:
- **Dataset**: BOLD5000 (OpenNeuro) — fMRI, ROIs release (CSI1-4), estímulos COCO/ImageNet/Scene.
- **Encoder semántico**: `openai/clip-vit-large-patch14` (target del backbone).
- **Generador**: `diffusers/stable-diffusion-2-1-unclip-i2i-l` en bf16 + xformers.
- **Adapter**: Red neuronal profunda (MindEye) con pérdida híbrida (InfoNCE + MSE + Cosine).

## Estado del proyecto

Rama actual: `main`. Migración completa de Ridge lineal a arquitectura profunda inspirada en MindEye. Estructura de directorios refactorizada a estándares de ingeniería.

**Aviso de Datos (Límite GitHub 5GB):**
Debido a las restricciones de tamaño de GitHub, los datos de fMRI y los estímulos no se suben al repositorio. Debes descargarlos manualmente y extraerlos en la raíz del proyecto:
- **BOLD5000 ROIs (.mat)**: [Descargar desde Figshare](https://figshare.com/articles/dataset/BOLD5000/6459449?file=12965447). Extraer en `BOLD5000_ROIs/`.
- **BOLD5000 Stimuli**: [Descargar desde Dropbox](https://www.dropbox.com/s/5ie18t4rjjvsl47/BOLD5000_Stimuli.zip?dl=1). Extraer en `BOLD5000_Stimuli/`.

## Requisitos

- Python 3.12
- NVIDIA GPU ≥ 8 GB VRAM (RTX 3070 mínimo, RTX 4070 Ti recomendada para entrenamiento masivo)
- CUDA 12.1
- ~15 GB libres (pesos SD 2.1 unCLIP + cache HF)

## Estructura del proyecto

```text
IA-ACECOM/
├── src/                          # Código fuente
│   ├── config.py                 # Paths unificados dinámicos (Local/RTX4070Ti)
│   ├── evaluation.py             # Métricas (SSIM, PixCorr, PSNR, LPIPS, pairwise)
│   ├── phase2_run_sd.py          # Inferencia SD 2.1 unCLIP desde embeddings
│   ├── sd_decoder.py             # Pipeline core diffusers
│   └── phase2/                   # Entrenamiento del modelo fMRI-a-CLIP
│       ├── mindeye_models.py     # Arquitectura de la red neuronal y loss (InfoNCE)
│       ├── train_mindeye.py      # Bucle de entrenamiento y dataloaders
│       └── visual_evaluator.py   # Pipeline E2E
├── docs/                         # Documentación e hitos
│   ├── arquitectura_sistema.md
│   ├── Hoja_de_Ruta_Tesis.md
│   ├── INTERFACE.md
│   └── MIGRATION.md
├── exe/                          # Ejecutables
│   └── ejecutable.bat            # Script maestro para generar reconstrucciones
└── README.md
```

## Ejecución

1. **Entrenar el modelo fMRI-a-CLIP (MindEye):**
   ```bash
   python -m src.phase2.train_mindeye --subject CSI1 --epochs 150 --batch_size 64
   ```

2. **Generar imágenes (Evaluación Visual E2E):**
   ```cmd
   exe\ejecutable.bat
   ```

## Créditos

Tesis de pregrado UNI (2025-2026), Alvaro Taipe Cotrina · Grupo ACECOM.
Inspiración arquitectónica y referencial: [MindEye: Reconstructing the Mind's Eye](https://medarc-ai.github.io/mindeye/) (MedARC-AI, Scotti et al. 2023/2024). Base teórica complementaria: Koide-Majima et al. (2024), Takagi-Nishimoto (2023).
