# Reconstrucción de Imágenes Mentales desde fMRI (Stable Diffusion 2.1 unCLIP + BOLD5000)

Proyecto de tesis (UNI · ACECOM) que decodifica imágenes desde actividad cerebral fMRI mapeando vóxeles al espacio semántico **CLIP** y decodificándolo con **Stable Diffusion 2.1 unCLIP**. El adapter fMRI→CLIP es intercambiable: un baseline **Ridge** (evaluado) y un backbone profundo tipo **MindEye** (peldaño de contribución).

## Arquitectura

```text
fMRI (BOLD5000 ROIs, ~10k vóxeles)
   │
   ▼
[Adapter fMRI→CLIP]  ──► z_CLIP ∈ R^768
   ├─ Ridge (solución cerrada)         → baseline evaluado
   ├─ Ridge estocástico (SGD)          → variante regularizable
   └─ MindEye MLP residual (InfoNCE)   → adapter profundo
   │
   ▼
SD 2.1 unCLIP UNet (frozen) + VAE decoder (frozen)
   │
   ▼
Imagen reconstruida
```

**Motivación:** el baseline Ridge colapsa parte de la variabilidad semántica en el espacio CLIP (texto alucinado, formas genéricas), y sus métricas cuantifican esa brecha frente a los pipelines de difusión guiada por CLIP de la literatura. Sobre esa base se construyen las variantes estocástica y profunda (InfoNCE + MSE + Cosine) para acercar los vectores fMRI a la métrica real del espacio latente visual.

Componentes:
- **Dataset**: BOLD5000 (OpenNeuro) — fMRI, ROIs release (CSI1-4), estímulos COCO/ImageNet/Scene.
- **Encoder semántico**: `openai/clip-vit-large-patch14` (target del adapter).
- **Generador**: `diffusers/stable-diffusion-2-1-unclip-i2i-l` en bf16.
- **Adapters**: `adapter_ridge.py`, `adapter_ridge_stoch.py`, `mindeye_models.py`.

## Estado del proyecto

Rama actual: `main`. Pipeline endurecido y portable (rutas `ACECOM_*`), suite de tests mock sin GPU, comparador cross-subject y galería de apéndices automatizada. Baseline Ridge evaluado sobre los 4 sujetos; adapter profundo como línea de trabajo siguiente.

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
│   ├── config.py                 # Paths/params centralizados (ACECOM_*)
│   ├── evaluation.py             # Métricas (SSIM, PixCorr, PSNR, LPIPS, CLIP, pairwise)
│   ├── extract_metrics.py        # Reporte R2/Cosine/MSE por sujeto
│   ├── locate_recons.py          # Resuelve rutas de salida + comando de empaquetado
│   ├── phase2_run_sd.py          # Inferencia SD 2.1 unCLIP desde embeddings
│   ├── sd_decoder.py             # Pipeline core diffusers
│   └── phase2/                   # Adapter fMRI→CLIP y evaluación
│       ├── bold5000_loader.py         # Carga betas ROI + alineación de stems
│       ├── extract_vit_features.py    # Targets CLIP ViT-L/14
│       ├── adapter_ridge.py           # Adapter Ridge (solución cerrada)
│       ├── adapter_ridge_stoch.py     # Ridge estocástico (SGD)
│       ├── mindeye_models.py          # Backbone MLP residual + InfoNCE
│       ├── train_adapter.py           # Entrena Ridge → embeds_test.pt
│       ├── train_mindeye.py           # Entrena MindEye
│       ├── visual_evaluator.py        # E2E: embeds → SD → GT|Recon + grid
│       ├── compare_subjects.py        # Figura [GT|CSI1..CSI4] por estímulo
│       └── build_appendix_montages.py # Galería paginada doble columna (Apéndice B)
├── tests/                        # Suite mock (pytest, sin GPU)
├── docs/                         # Documentación e hitos
├── exe/                          # Ejecutables (ejecutable.bat)
└── AlvaroTaipe_Plantilla/        # Fuentes LaTeX de la tesis
```

## Ejecución

Todos los comandos se corren desde `src/`. Las rutas se resuelven vía `config`
y variables `ACECOM_*` (ver `src/config.py`).

```bash
cd src
# 1. Targets CLIP ViT-L/14 de los estímulos presentados
python -m phase2.extract_vit_features

# 2. Adapter fMRI→CLIP (por sujeto) → embeds_test.pt
python -m phase2.train_adapter --subject CSI1        # o train_mindeye

# 3. Reconstrucción SD 2.1 unCLIP + collages GT|Recon
python -m phase2.visual_evaluator --subject CSI1

# 4. Galería cross-subject paginada (Apéndice B)
python -m phase2.build_appendix_montages --rows-per-page 12 --per-page 2 --emit-tex

# Utilidad: ¿dónde quedaron las reconstrucciones?
python locate_recons.py
```

Alternativa Windows: `exe\ejecutable.bat` orquesta la reconstrucción.
Tests sin GPU: `pytest` desde la raíz.

## Créditos

Tesis de pregrado UNI (2025-2026), Alvaro Taipe Cotrina · Grupo ACECOM.
Inspiración arquitectónica y referencial: [MindEye: Reconstructing the Mind's Eye](https://medarc-ai.github.io/mindeye/) (MedARC-AI, Scotti et al. 2023/2024). Base teórica complementaria: Koide-Majima et al. (2024), Takagi-Nishimoto (2023).
