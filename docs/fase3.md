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

### 2.2 Norma del embedding configurable
`src/phase2/visual_evaluator.py`: `--embed-norm {ridge,unit,none}` + `--embed-scale`.

El `F.normalize(emb)*12.0` hardcodeado era un parche para el *shrinkage* de
Ridge. Para embeds de MindEye, cuya magnitud ya está anclada por el término MSE
de la pérdida, ese reescalado borra información per-muestra que `noise_level`
de unCLIP debería modular.

```bash
# MindEye: dejar la norma cruda
py -3.12 -m phase2.visual_evaluator --subject CSI1 --embed-norm none
```

> Ablation pendiente en Máquina B: `ridge` vs `none` × `noise_level ∈ {0,15}`.
> Reportar pairwise ID y LPIPS por celda.

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
