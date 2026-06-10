# Fase 3 — Endurecimiento del código y mejoras de decodificación

**Objetivo de la fase:** dejar el pipeline reproducible, testeado y comparable
*antes* de tocar la redacción de la tesis. No se modifica la plantilla LaTeX
hasta tener resultados nuevos; primero mejoramos el código.

Contexto cuantitativo (baseline a superar): pairwise ID ≈ **0.495** (azar) con
Ridge lineal; Koide-Majima 2024 reporta **0.756** en imagery. Ver
`docs/STATE.md` y la nota de baseline.

---

## 1. Bugs corregidos

| Archivo | Bug | Efecto | Fix |
|---|---|---|---|
| `src/evaluation.py` | `parse_args()` leía `config.NSD_CONFIG` (removido en el pivote a BOLD5000) | El script crasheaba al invocarse por CLI | Reescrito sobre `config.BOLD5000_CONFIG` / `config.BOLD5000_SUBJECTS` |
| `src/evaluation.py` | `load_reconstructions` esperaba `{subject}_{label}_sd_unclip.png` | No casaba con lo que escribe `visual_evaluator.py` (`{stem}_recon.png`) → 0 pares evaluables | Lee `{recon}/{subject}/reconstructions/{stem}_recon.png`; GT por `stem` vía rglob (COCO/ImageNet/Scene) |
| `src/phase2/mock_data.py` | `clip_*` salía `float64` (upcast por `.std()`) | Violaba el contrato `Split` (float32) | Cast explícito a `float32` |

Cobertura de regresión: `tests/test_evaluation_metrics.py`,
`tests/test_loader_mock.py`.

---

## 2. Mejoras de decodificación (implementadas, opt-in)

### 2.1 Schedule contrastivo BiMixCo → SoftCLIP (MindEye)
`src/phase2/mindeye_models.py` + `train_mindeye.py`.

- `mixco_sample` + `mixco_nce`: mixup de vóxeles con InfoNCE de etiquetas
  mezcladas (primer tercio del entrenamiento).
- `soft_clip_loss`: etiquetas suaves target↔target (dos tercios finales).
- Activación: `--mixco` (off por defecto, reproduce el baseline actual).

```bash
py -3.12 -m phase2.train_mindeye --subject CSI1 --epochs 150 --batch_size 64 --mixco
```

**Por qué:** es el lift de *pairwise ID* documentado por Scotti et al. en
régimen de pocas muestras — exactamente BOLD5000 (~4.8k trials/sujeto). Todo el
cómputo de logits va en float32 (estable en bf16, sin `exp()` no acotado).

### 2.2 Norma del embedding configurable — **default cambiado a `none`**
`src/phase2/visual_evaluator.py`: `--embed-norm {ridge,unit,none}` + `--embed-scale`.

**Por qué `none` es ahora el default.** `extract_vit_features.py` guarda los
targets CLIP **sin L2-normalizar** (`image_embeds` crudos). MindEye ancla su
salida a esa magnitud (término MSE), y SD 2.1 unCLIP espera recibir el
`image_embeds` CLIP en su escala cruda (su `image_normalizer` ya resta media /
divide std internamente). El `F.normalize(emb)*12.0` hardcodeado (parche para el
*shrinkage* de Ridge) **distorsionaba esa escala** y degradaba la reconstrucción.

```bash
# MindEye (default): norma cruda
py -3.12 -m phase2.visual_evaluator --subject CSI1
# Legacy adapter Ridge (norma aplastada): restaurar escala
py -3.12 -m phase2.visual_evaluator --subject CSI1 --embed-norm ridge
```

> Ablation pendiente en Máquina B: `none` vs `ridge` × `noise_level ∈ {0,15}`.
> Reportar pairwise ID y LPIPS por celda.

### 2.3 Robustez de carga SD 2.1 unCLIP
`src/sd_decoder.py`: `load_sd_unclip_pipeline` ya no asume que el repo publica
pesos `variant="fp16"`. Si la carga con variant falla, reintenta sin variant
(evita un crash duro en el primer arranque de la Máquina B).

### 2.4 Logs ASCII-safe (Windows cp1252)
Los glyphs `→`/`★` en mensajes de log/argparse provocaban `UnicodeEncodeError`
en consolas cp1252 (Windows) y abortaban la reconstrucción a media corrida.
Reemplazados por ASCII (`->`, `*`) en todos los scripts del flujo. Además
`exe/ejecutable.bat` fija `chcp 65001` + `PYTHONIOENCODING=utf-8`.

---

## 3. Mejoras priorizadas pendientes (orden por costo/beneficio)

1. **Promediado de repeticiones en TEST ya activo** (`bold5000_loader.py` promedia
   las ~4 reps de los 113 repeated). Verificar que el conteo de reps por stem es
   el esperado; subir SNR sumando reps de train para los estímulos que las tengan.
2. **Prior de difusión ligero** (MindEye reconstruction submodule): MLP que mapea
   `z_fMRI → z_CLIP_img` minimizando en el espacio del prior, en vez de feedear el
   embed crudo a unCLIP. Mayor fidelidad estructural. Costo medio; cabe en 12 GB.
3. **Inicialización low-level (Brain-Diffuser)**: predecir un latente VDVAE/VAE de
   baja resolución como `init_image` de unCLIP (img2img) → ataca PixCorr≈0.
4. **Decodificador multimodal (texto+imagen)**: añadir un target de caption CLIP.
   NSD-Imagery (CVPR 2025) muestra que la decodificación multimodal generaliza
   mejor a *imagery* — el objetivo final de esta tesis.

Detalle y referencias por técnica: `docs/RAG_reconstruccion_fmri.md`.

---

## 4. Comparación cross-subject (4 individuos)

`src/phase2/compare_subjects.py` — figura única con `[GT | CSI1 | CSI2 | CSI3 | CSI4]`
sobre el MISMO estímulo (los 113 repeated son compartidos).

```bash
# Panel agregado (8 estímulos comunes a los 4 sujetos)
py -3.12 -m phase2.compare_subjects --limit 8

# Una sola muestra en detalle
py -3.12 -m phase2.compare_subjects --stem desertvegetation3 --dpi 200
```

Entrada: `{eval_dir}/{subject}/reconstructions/{stem}_recon.png` (lo que produce
`visual_evaluator`). Sin torch/diffusers: corre en Máquina A o B tras `git pull`.

---

## 5. Testing (Máquina A, sin GPU)

```bash
py -3.12 -m pytest        # 24 tests, ~13 s
```

- `test_reconstruction_dryrun.py` — pipeline E2E completo con `--dry-run` (stub PIL
  en vez de SD 2.1). Valida IO, alineación stems↔embeds y render sin gastar GPU.
- `test_mindeye_models.py` — shapes, claves de pérdida, gradiente, clamp bf16.
- `test_compare_subjects.py` — intersección de sujetos + celdas faltantes.
- `test_evaluation_metrics.py` — métricas + el wiring que estaba roto.
- `test_loader_mock.py` — contrato `Split`.

**Flujo de trabajo Máquina A → B:** se codifica y testea aquí (mock, CPU), se
`git push`; la Máquina B (RTX 4070 Ti) hace `git pull` y corre la reconstrucción
real (`exe/ejecutable.bat`) + `compare_subjects` + `evaluation`.

---

## 6. Escalera de ablación (corrida final en RTX 4070 Ti)

Narrativa de publicación: **Ridge Lineal → Ridge Estocástico → MindEye**, midiendo
cuánto cierra el Modality Gap por unidad de cómputo. Estas tres corridas llenan
las Tablas 7.1–7.4 del documento (`AlvaroTaipe_Plantilla/main.tex`), hoy en TBD.

| Rung | Módulo | Embeds | Costo |
|---|---|---|---|
| Ridge Lineal (diagnóstico) | `phase2/train_adapter.py` | crudos | cerrado, segundos |
| **Ridge Estocástico (contribución)** | `phase2/adapter_ridge_stoch.py` | `ê=Xβ+σξ`, renorm a esfera | +1 hiperparam σ |
| MindEye (techo) | `phase2/train_mindeye.py` | MLP residual + InfoNCE | entrenamiento profundo |

Protocolo por sujeto (CSI1–CSI4):

```bash
# 0) targets CLIP (una vez) + Ridge lineal
py -3.12 -m phase2.extract_vit_features --stimuli-dir <stimuli> --out phase2_outputs/clip_targets/bold5000_vitL14.pt
py -3.12 -m phase2.train_adapter      --mode bold5000 --subject CSI1   # -> adapter/CSI1/embeds_test.pt
# 1) Ridge Estocástico (calibra σ por pairwise en validación)
py -3.12 -m phase2.adapter_ridge_stoch --mode bold5000 --subject CSI1  # -> adapter_stoch/CSI1/embeds_test.pt
# 2) MindEye
py -3.12 -m phase2.train_mindeye      --subject CSI1 --epochs 150 --mixco  # -> mindeye/CSI1/embeds_test.pt

# 3) Reconstrucción por rung (un eval-dir distinto por variante; --embed-norm none: el adapter ya normaliza)
py -3.12 -m phase2.visual_evaluator --subject CSI1 --embeds phase2_outputs/adapter/CSI1/embeds_test.pt        --out-dir out_ridge       --embed-norm none
py -3.12 -m phase2.visual_evaluator --subject CSI1 --embeds phase2_outputs/adapter_stoch/CSI1/embeds_test.pt  --out-dir out_stoch       --embed-norm none
py -3.12 -m phase2.visual_evaluator --subject CSI1 --embeds phase2_outputs/mindeye/CSI1/embeds_test.pt        --out-dir out_mindeye     --embed-norm none

# 4) Métricas por rung (llena Tablas 7.1–7.3) y comparador cross-subject (Fig. agregada)
py -3.12 -m evaluation --subjects CSI1 CSI2 CSI3 CSI4 --recon-dir out_stoch
py -3.12 -m phase2.compare_subjects --eval-dir out_stoch --limit 8
```

**Nota de norma:** el Ridge Estocástico renormaliza a la hiperesfera unidad por
construcción (eq 6.3), así que `visual_evaluator --embed-norm none` evita
doble-normalizar. Si la norma unidad resulta demasiado pequeña para unCLIP
(síntoma: salidas genéricas), ablar con `adapter_ridge_stoch --embed-scale 12`
(coloca el vector en la escala cruda de CLIP ViT-L/14). Reportar ambos en la tesis.
