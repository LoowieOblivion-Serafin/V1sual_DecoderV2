# Arquitectura de Software: V1sual_DecoderV2 (Fase 2)

El sistema fue diseñado con el patrón de **Arquitectura de Tubería (Pipeline)**. Esto significa que cada archivo Python hace *una sola cosa muy bien* y le pasa el resultado al siguiente. Esto evita que el sistema colapse y facilita futuras actualizaciones.

## Diagrama de Flujo (Pipeline)

```mermaid
graph TD
    %% Archivos de Configuración
    Config[config.py\n(Control Central y Rutas)] --> Loader
    Config --> Extractor
    Config --> Trainer
    Config --> Evaluator
    Config --> Decoder

    %% Fase 1: Extracción de Datos
    subgraph Fase 1: Preprocesamiento
        Loader[phase2/bold5000_loader.py\n(Lee los .mat del cerebro)] --> X_train[(Matrices X\nfMRI Vóxeles)]
        Extractor[phase2/extract_vit_features.py\n(Lee fotos y pasa por CLIP)] --> Y_train[(Vectores Y\nCLIP Embeddings)]
    end

    %% Fase 2: Entrenamiento
    subgraph Fase 2: El Puente Neural
        X_train --> Trainer[phase2/train_adapter.py\n+ adapter_ridge.py]
        Y_train --> Trainer
        Trainer --> Pesos[(Pesos Entrenados\nModelos Ridge)]
    end

    %% Fase 3: Inferencia y Generación
    subgraph Fase 3: Inferencia (Evaluación)
        Pesos --> Evaluator[phase2/visual_evaluator.py\n(Orquestador)]
        Loader --> |fMRI de Prueba Inéditos| Evaluator
        Evaluator --> |Vector Predicho de 768-D| Decoder[sd_decoder.py\n(Stable Diffusion unCLIP)]
        Decoder --> |Imagen Generada| Evaluator
        Evaluator --> Resultados[(output_reconstruccions_test2\nGrids Comparativos)]
    end
```

## Anatomía de los Módulos Principales

Aquí tienes la radiografía exacta de qué hace cada archivo importante que construimos. Si a futuro quieres actualizar o mejorar algo, aquí es donde debes mirar:

### 1. El Cerebro del Código: `config.py`
- **¿Qué hace?** Es el único lugar donde existen variables globales, rutas a carpetas y parámetros matemáticos (como los 75 pasos de `num_inference_steps` que ajustamos).
- **Actualización futura:** Si mañana quieres usar otra versión de Stable Diffusion, o si descargas más datos, **solo cambias este archivo**. Ningún otro script necesita ser tocado.

### 2. El Lector de Mentes: `phase2/bold5000_loader.py`
- **¿Qué hace?** Va a la carpeta pesada `BOLD5000_ROIs`, lee los archivos `.mat` de Matlab (que son un desastre biológico) y los convierte en hermosas matrices NumPy estructuradas y limpias.

### 3. El Compresor de Imágenes: `phase2/extract_vit_features.py`
- **¿Qué hace?** Toma las fotografías originales que el paciente vio en la resonancia, se las da al modelo de OpenAI (`CLIP ViT-L/14`) y guarda la esencia de la imagen en vectores matemáticos (`.pt`). 

### 4. El Entrenador Matemático: `phase2/train_adapter.py` y `adapter_ridge.py`
- **¿Qué hace?** Aquí reside la Inteligencia Artificial tradicional. Coge lo que salió del Lector de Mentes ($X$) y del Compresor de Imágenes ($Y$) y entrena un modelo estadístico (Regresión Ridge) para mapear el cerebro a los vectores. **Guarda un archivo de "pesos" por cada paciente**.
- **Actualización futura:** Si mañana quieres cambiar Ridge por una Red Neuronal Profunda o un Autoencoder, solo modificarás estos dos archivos.

### 5. El Generador de Arte: `sd_decoder.py`
- **¿Qué hace?** Simplemente levanta Stable Diffusion 2.1 unCLIP. Recibe un vector numérico de 768 dimensiones y devuelve un cuadro (imagen). No sabe de dónde vino el vector (no sabe si es del cerebro o inventado).
- **Actualización futura:** Si en un año sale Stable Diffusion 3.0 unCLIP o un modelo mejor, solo tocas este archivo.

### 6. El Capataz (Orquestador): `phase2/visual_evaluator.py`
- **¿Qué hace?** Es el script que une todo al final. Es el que llama el archivo `.bat`. Lo que hace es: "Trae los pesos entrenados del Paciente 1, trae su fMRI de prueba, cálcula el vector predicho, dáselo al generador de arte (`sd_decoder.py`), y guarda la imagen original y la nueva en un collage lado a lado". Además, calcula la Similitud Coseno.
