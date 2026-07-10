# Delimitación Epistemológica y Estructural de la Tesis

## El problema real no es técnico, sino de direccionamiento

Actualmente la investigación posee tres ejes legítimos:

1. **Ciencias de la Computación**
   - Complejidad computacional.
   - Optimización.
   - Espacios latentes.
   - Selección de vóxeles.
   - Formalización matemática.
   - Procesamiento de alta dimensionalidad.

2. **Inteligencia Artificial**
   - CLIP.
   - Stable Diffusion.
   - Diffusion Priors.
   - Embeddings semánticos.
   - Aprendizaje contrastivo.
   - Modelos generativos multimodales.

3. **Neurociencia Computacional**
   - Señal BOLD.
   - ROIs.
   - Jerarquía cortical visual.
   - Codificación semántica visual.
   - Organización V1/V2/V4/HVC.

El problema es que actualmente los tres aparecen como líneas paralelas, cuando en realidad deben subordinarse a un único hilo conductor.

---

# El verdadero núcleo de la tesis

La tesis NO es sobre:

- Stable Diffusion.
- CLIP.
- NP-complete.
- Neurociencia general.
- Reconstrucción visual en abstracto.

La tesis realmente trata sobre:

> **La traducción intermodal entre actividad cerebral visual y espacios semánticos latentes mediante adapters contrastivos y modelos generativos condicionados.**

Todo lo demás debe actuar únicamente como:

- fundamento,
- herramienta,
- soporte metodológico,
- o consecuencia.

---

# Reorganización conceptual correcta

## 1. Neurociencia Computacional = Dominio Fuente

Esta sección responde:

> “¿Qué información existe en el cerebro y cómo está estructurada?”

Debe cubrir:

- Señal BOLD.
- Organización cortical.
- Jerarquía visual.
- ROIs.
- Representación neuronal distribuida.
- Problema del ruido en fMRI.

Aquí NO deben dominar:
- Stable Diffusion.
- CLIP.
- Arquitecturas generativas.

---

## 2. Inteligencia Artificial = Mecanismo de Traducción

Esta sección responde:

> “¿Cómo traducimos actividad cerebral a representaciones semánticas utilizables?”

Debe cubrir:

- CLIP.
- Espacios latentes.
- Embeddings multimodales.
- Diffusion Priors.
- Stable Diffusion.
- Aprendizaje contrastivo.
- Alineamiento multimodal.

---

## 3. Ciencias de la Computación = Formalización del Problema

Esta sección responde:

> “¿Cómo resolvemos computacionalmente el problema bajo restricciones reales?”

Debe cubrir:

- Complejidad computacional.
- Optimización.
- Regularización.
- Selección de vóxeles.
- Dimensionalidad \( p \gg n \).
- Complejidad combinatoria.
- Restricciones de VRAM y entrenamiento.

---

# Advertencia importante sobre NP-Completeness

El problema de 3-Partition NO debe convertirse en el eje central de la tesis.

Debe utilizarse únicamente como:

- justificación teórica,
- argumento de complejidad,
- o motivación metodológica.

Ejemplo correcto:

> “La selección óptima de subconjuntos de vóxeles puede formularse como un problema combinatorio intratable; por ello se emplean heurísticas biológicas y regularización Ridge.”

Ejemplo incorrecto:

> “Esta tesis resuelve un problema NP-hard…”

Eso desviaría la identidad de la investigación y podría percibirse como sobreextensión conceptual.

---

# Prioridades reales para Tesis 1

El objetivo de Tesis 1 NO es demostrar estado del arte.

El objetivo es demostrar:

1. Delimitación clara del problema.
2. Comprensión profunda del estado del arte.
3. Coherencia metodológica.
4. Viabilidad técnica.
5. Hipótesis científicamente defendible.

---

# Estructura recomendada de los capítulos principales

## Resumen

Debe responder únicamente:

- Qué problema.
- Qué arquitectura.
- Qué dataset.
- Qué metodología.
- Qué hipótesis.
- Qué contribución.

Debe evitar:
- Filosofía extensa.
- Microdetalles técnicos.
- Sobreexplicación matemática.

---

## Introducción

### Estructura ideal

1. Problema global.
2. Limitaciones actuales.
3. Estado moderno del área.
4. Problema específico.
5. Hipótesis.
6. Contribución.

---

## Estado del Arte

Debe organizarse cronológicamente:

### Primera generación
- VGG.
- Deep Image Reconstruction.
- Bayesian Decoding.

### Segunda generación
- CLIP.
- Brain-Diffuser.
- Versatile Diffusion.

### Tercera generación
- MindEye.
- MindEye2.
- NeuroPictor.
- HI-DREAM.

### Gap actual
El adapter semántico fMRI→CLIP sigue siendo el cuello de botella.

---

## Herramientas

Este capítulo NO debe parecer una lista de librerías.

Debe justificar:

- Por qué PyTorch.
- Por qué CLIP ViT-L/14.
- Por qué Stable Diffusion 2.1.
- Por qué BOLD5000 o NSD.
- Por qué bf16/xformers.
- Por qué Ridge o adapters contrastivos.

---

# Identidad académica correcta de la tesis

> “Esta investigación se sitúa en la intersección entre neurociencia computacional, aprendizaje profundo multimodal y representación semántica contrastiva, enfocándose específicamente en el problema del alineamiento entre señales fMRI y espacios latentes CLIP para reconstrucción visual.”