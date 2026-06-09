# RAG — Reconstrucción de imágenes desde fMRI: estado del arte mundial

> **Propósito.** Base de conocimiento curada de los proyectos de reconstrucción
> visual desde fMRI (2019–2026) para **mejorar los resultados de ESTE código**.
> Cada técnica se mapea a archivos concretos del repo. Es el insumo de
> `docs/fase3.md` y complementa la delimitación conceptual de
> `AlvaroTaipe_Plantilla/RAG_gpt.md`.
>
> Leyenda de confianza: ✅ verificado vía búsqueda web (2024–2026) · 📚 base
> teórica consolidada (papers en `Papers/`).
>
> Stack actual del repo: **fMRI (BOLD5000 ROIs) → MindEyeBackbone (MLP residual,
> InfoNCE) → CLIP ViT-L/14 (768-d) → SD 2.1 unCLIP (frozen)**.

---

## 0. Tabla resumen

| Proyecto | Año | Dataset | Generador | Técnica clave | Métrica destacada |
|---|---|---|---|---|---|
| Shen et al. (iCNN) | 2019 | Diseño propio | DGN + optimización | Iterativo sobre features jerárquicas | Cualitativo |
| Takagi–Nishimoto | 2023 | NSD | Stable Diffusion | Reg. lineal a latente + CLIP texto | SSIM/PixCorr altos |
| **Brain-Diffuser** (Ozcelik) 📚 | 2023 | NSD | VDVAE + Versatile Diffusion | **2 etapas**: low-level VDVAE → high-level multimodal | SOTA 2023 |
| **MindEye** (Scotti) ✅ | 2023 | NSD | Versatile Diffusion | Contrastivo (**BiMixCo**) + **diffusion prior** | Retrieval casi perfecto |
| **MindEye2** ✅ | 2024 | NSD | SDXL unCLIP | **Shared-subject** + 1 h de datos | **SOTA casi todas las métricas** |
| MindBridge ✅ | 2024 | NSD | LDM | Cross-subject con un solo modelo | Generaliza entre sujetos |
| NeuroPictor ✅ | 2024 | NSD (67k pares) | Diffusion + ControlNet | Pre-entreno multi-individuo + **modulación multinivel** | Fuerte within-subject |
| Lite-Mind | 2024 | NSD | — (retrieval) | Backbone DFT eficiente, pocos params | Retrieval barato |
| UMBRAE | 2024 | NSD | MLLM | Brain encoder universal + captioning | Multimodal |
| Koide-Majima 📚 | 2024 | Diseño propio | Bayesiano + Langevin | **Imagery**; CLIP+VGG, prior bayesiano | Pairwise 75.6% imagery |
| **NSD-Imagery** (Kneeland) ✅ | 2025 | NSD-Imagery | (benchmark) | **Lineal+multimodal generaliza mejor a imagery** | Benchmark imagery |
| MIRAGE ✅ | 2025 | NSD | Diffusion | Arquitecturas robustas visión→imagery | Imagery |
| Dynadiff ✅ | 2025 | fMRI continuo | Diffusion single-stage | Decode de fMRI que evoluciona en el tiempo | — |
| BrainCognizer / Brain-IT ✅ | 2025 | NSD | Diffusion / Transformer | Simulación de cognición visual / interacción cerebral | Frontera |

---

## 1. Fundacionales

### Shen et al. 2019 — *Deep image reconstruction* 📚
Optimización iterativa de una imagen para que sus features (CNN jerárquica)
casen con las decodificadas del fMRI, regularizada por un *deep generator
network*. Es el ancestro del enfoque "optimiza el latente" del baseline VGG+VQGAN
ya purgado. **Lección:** la optimización directa sobre el latente produce
artefactos (el problema que motivó migrar a MindEye). No volver a ese enfoque.

### Takagi & Nishimoto 2023 — *Stable Diffusion + NSD* 📚
Regresión **lineal** de fMRI a (a) latente VAE de SD y (b) embedding CLIP de
texto. Simple y sorprendentemente fuerte en métricas de bajo nivel.
**Para el repo:** confirma que una rama lineal sólida debe existir como
*baseline honesto* y como red de seguridad para imagery (ver §4 NSD-Imagery).

### Brain-Diffuser (Ozcelik & VanRullen) 2023 — `Papers/s41598-023-42891-8.pdf` 📚 ✅
Pipeline de **dos etapas**:
1. **Low-level:** fMRI → latente **VDVAE** → imagen de baja resolución que
   captura layout y color global.
2. **High-level:** esa imagen entra como `init_image` a **Versatile Diffusion**
   (img2img) condicionada por features multimodales (texto+visión) predichas.

> **Adoptar (alta prioridad).** El `PixCorr ≈ 0` del baseline indica que NO se
> recupera estructura espacial. Predecir un `init_image` low-level y usar el modo
> **img2img** de SD 2.1 unCLIP ataca esto directamente.
> Archivos: nuevo `phase2/lowlevel_decoder.py` + `strength`/`init_image` en
> `sd_decoder.reconstruct_from_embedding`.
> Repo de referencia: `github.com/ozcelikfu/brain-diffuser`.

---

## 2. Contrastivo + diffusion prior (la familia de este repo)

### MindEye (Scotti et al., NeurIPS 2023) — `Papers/2305.18274v2.pdf` ✅
Dos submódulos paralelos desde el embedding de cerebro:
- **Retrieval:** aprendizaje **contrastivo** (CLIP loss) con **BiMixCo**
  (mixup bidireccional). Schedule: primer 1/3 BiMixCo con etiquetas duras, 2/3
  finales SoftCLIP con etiquetas suaves. Logra recuperar la imagen exacta entre
  candidatas muy similares.
- **Reconstruction:** un **diffusion prior** (entrenado desde cero) que alinea el
  embedding de cerebro con el espacio CLIP de imagen *antes* de pasarlo al
  generador (Versatile Diffusion).

> **Ya adoptado:** `MindEyeBackbone` + `MindEyeLoss` (InfoNCE bidireccional) y,
> en Fase 3, **BiMixCo→SoftCLIP** (`mixco_sample`/`mixco_nce`/`soft_clip_loss`,
> flag `--mixco`). Ver `docs/fase3.md §2.1`.
> **Pendiente de adoptar:** el **diffusion prior** como puente
> `z_fMRI → z_CLIP_img` (en lugar de feedear el embed crudo a unCLIP). Es la
> mejora de fidelidad más alineada con la arquitectura actual.

### MindEye2 (ICML 2024) — `Papers/MindEye2.pdf` y `2403.11207v2.pdf` ✅
- **Shared-subject model:** pre-entrena en 7 sujetos en un espacio latente común
  y hace fine-tuning con **1 hora** de datos de un sujeto nuevo (2.5% de los
  datos). **SOTA en casi todas las métricas** con NSD completo.
- Usa **SDXL unCLIP** y mapea a un espacio CLIP de mayor capacidad.

> **Adoptar (media prioridad, encaja con los 4 sujetos BOLD5000):** un
> **encoder compartido** CSI1–CSI4 con cabezas ridge por-sujeto (alinear los
> espacios de vóxeles) multiplica los datos efectivos. Es la palanca natural del
> comparador cross-subject (`phase2/compare_subjects.py`).
> Archivos: refactor de `train_mindeye.py` a multi-sujeto + proyección de entrada
> por sujeto. Repo: `github.com/MedARC-AI/MindEyeV2`.

---

## 3. Cross-subject y multi-individuo

### MindBridge 2024 ✅
Un único modelo para varios sujetos; reduce el *gap* de modalidad usando datos
multi-sujeto para mejorar la generalización individual.

### NeuroPictor 2024 (ECCV) — `Papers/2403.11207`... (ver arXiv 2403.18211) ✅
Pre-entrenamiento **multi-individuo** (~67k pares) + **modulación multinivel**:
features semánticos de alto nivel para el "qué", y una red de manipulación de
bajo nivel para instrucciones estructurales finas sobre el modelo de difusión.

> **Adoptar:** la idea de **dos niveles** (semántico + estructural) coincide con
> Brain-Diffuser y refuerza la prioridad del `init_image` low-level (§1).

### Lite-Mind 2024 · UMBRAE 2024
- **Lite-Mind:** backbone eficiente (transformada de Fourier) para retrieval con
  muy pocos parámetros — interesante para la **RTX 2070 (8 GB)** de desarrollo.
- **UMBRAE:** brain encoder universal acoplado a un **MLLM** para captioning;
  línea multimodal.

---

## 4. Imagery / imágenes mentales (el objetivo final de la tesis)

### Koide-Majima & Nishimoto 2024 — `Papers/Mental image reconstruction...pdf` 📚
Reconstrucción de **imaginería visual** (no sólo percepción) con un marco
**bayesiano** + dinámica de **Langevin** sobre features CLIP+VGG. Baseline de la
tesis: **pairwise 75.6%** en imagery. Es la meta a superar.

### NSD-Imagery (Kneeland et al., CVPR 2025) — `Papers/NSD_Imagery.pdf` ✅
Benchmark de fMRI ↔ imágenes **mentales**. Hallazgo central, **crítico para esta
tesis**:

> *"El desempeño en imágenes mentales está en gran medida desacoplado del
> desempeño en reconstrucción de percepción; los modelos con arquitecturas de
> decodificación **lineales** y **decodificación multimodal** generalizan mejor a
> la imaginería mental."*

**Implicación de diseño.** Optimizar la red profunda sólo para reconstruir
*imágenes vistas* puede **perjudicar** la generalización a imagery. Por eso:
1. Mantener una **rama lineal/ridge** evaluada en paralelo (no borrarla).
2. Añadir **target multimodal** (caption CLIP) además del embedding de imagen.
3. Reportar imagery y percepción **por separado**.

### MIRAGE 2025 ✅
Arquitecturas multimodales robustas que **trasladan** modelos visión→imagery.
Confirma la dirección multimodal de NSD-Imagery.

---

## 5. Frontera 2025–2026 ✅
- **Dynadiff:** decodificación *single-stage* de fMRI que evoluciona en el tiempo
  (relevante si se exploran TRs individuales en vez de TR34 promediado).
- **BrainCognizer / Brain-IT:** simulación de cognición visual humana y
  *Brain-Interaction Transformer*; mejoras sobre límites de métodos previos.

---

## 6. Técnicas transversales → mapeo a ESTE repo

| Técnica | Origen | Estado en repo | Acción concreta |
|---|---|---|---|
| InfoNCE bidireccional | MindEye | ✅ hecho | `mindeye_models.MindEyeLoss` |
| **BiMixCo → SoftCLIP** | MindEye | ✅ Fase 3 (opt-in) | `--mixco` en `train_mindeye` |
| Norma de embed cruda para unCLIP | MindEye | ✅ Fase 3 | `--embed-norm none` en `visual_evaluator` |
| Promediar repeticiones (test) | NSD/MindEye | ✅ hecho | `bold5000_loader` promedia 113 repeated |
| **Diffusion prior** (z_fMRI→z_CLIP) | MindEye | ⏳ pendiente | nuevo `phase2/diffusion_prior.py` |
| **Init low-level (VDVAE/VAE)** img2img | Brain-Diffuser | ⏳ pendiente | `phase2/lowlevel_decoder.py` + `strength` en `sd_decoder` |
| Decodificación **multimodal** (texto) | Takagi / NSD-Imagery | ⏳ pendiente | 2º target CLIP-texto en el loader/trainer |
| **Shared-subject** encoder | MindEye2 | ⏳ pendiente | `train_mindeye` multi-sujeto + cabeza por sujeto |
| Backbone eficiente (Fourier) | Lite-Mind | 💡 opción 8 GB | alternativa ligera del backbone |

Prioridad sugerida (costo/beneficio, RTX 4070 Ti 12 GB): **1)** `--mixco` +
`--embed-norm none` (ya disponibles, sólo correr ablation) → **2)** init
low-level img2img → **3)** diffusion prior → **4)** target multimodal → **5)**
shared-subject.

---

## 7. Métricas estándar del campo (para reportar en la tesis)

- **Bajo nivel:** PixCorr (correlación de píxeles), SSIM, AlexNet(2/5).
- **Alto nivel:** Inception, CLIP, EfficientNet-B1, SwAV.
- **Pairwise / 2-AFC identification:** ¿el embed reconstruido se parece más a su
  GT que a un distractor? Es la métrica *gold* (Scotti 2023, Koide-Majima 2024).
  Implementada en `train_mindeye.pairwise_accuracy` y `evaluation.pairwise_identification`.
- **Retrieval (top-1/top-k):** porcentaje de recuperación de la imagen exacta.

Reportar **siempre** percepción e imagery por separado (lección NSD-Imagery).

---

## 8. Fuentes

- [MindEye — Reconstructing the Mind's Eye (arXiv 2305.18274)](https://arxiv.org/abs/2305.18274) · [proyecto](https://medarc-ai.github.io/mindeye/)
- [MindEye2 (arXiv 2403.11207)](https://arxiv.org/abs/2403.11207) · [proyecto](https://medarc-ai.github.io/mindeye2/) · [código MedARC](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)
- [Brain-Diffuser (arXiv 2303.05334)](https://arxiv.org/abs/2303.05334) · [Scientific Reports](https://www.nature.com/articles/s41598-023-42891-8) · [código](https://github.com/ozcelikfu/brain-diffuser)
- [NeuroPictor (arXiv 2403.18211)](https://arxiv.org/abs/2403.18211)
- [Mind-Bridge (ResearchGate)](https://www.researchgate.net/publication/380347156)
- [NSD-Imagery (arXiv 2506.06898)](https://arxiv.org/abs/2506.06898) · [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Kneeland_NSD-Imagery_A_Benchmark_Dataset_for_Extending_fMRI_Vision_Decoding_Methods_CVPR_2025_paper.html)
- [MIRAGE (PLOS Comput Biol)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014263)
- [Dynadiff (arXiv 2505.14556)](https://arxiv.org/abs/2505.14556)
- [Brain-IT (arXiv 2510.25976)](https://arxiv.org/html/2510.25976v1) · [BrainCognizer (arXiv 2510.20855)](https://arxiv.org/html/2510.20855v1)

*Papers locales en `Papers/`: MindEye2, NSD_Imagery, Mental image reconstruction,
Natural scene reconstruction (Brain-Diffuser), 2305.18274 (MindEye), 2403.11207.*
