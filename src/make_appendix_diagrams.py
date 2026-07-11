"""
===============================================================================
make_appendix_diagrams.py — Diagramas vectoriales para los Apéndices A y C
===============================================================================

Genera, con matplotlib (sin dependencias de red ni graphviz), los diagramas que
consumen los apéndices imagen+caption:

    Figures/deepwiki/arquitectura_e2e.png       (Apéndice A)
    Figures/deepwiki/flujo_modulos_phase2.png   (Apéndice A)
    Figures/deepwiki/mapa_modulos_repo.png      (Apéndice A)
    Figures/deepwiki/proceso_difusion.png       (Apéndice C)
    Figures/deepwiki/ridge_vs_mindeye.png       (Apéndice C)

El contenido (rutas archivo:línea, flujo de módulos) proviene del Codemap de
DeepWiki sobre el repo actual, así que refleja el código real.

Uso:
    python make_appendix_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "AlvaroTaipe_Plantilla" / "Figures" / "deepwiki"
)

# Paleta sobria para impresión (fondo claro).
C = {
    "in":     ("#e8f0fe", "#1a56b0"),   # entrada fMRI
    "adapt":  ("#fff0e0", "#b5651d"),   # adapter
    "clip":   ("#e6f4ea", "#1e7d34"),   # CLIP
    "sd":     ("#f0e8fb", "#6b3fb5"),   # Stable Diffusion
    "out":    ("#e0f7f5", "#0d7d74"),   # salida
    "neutral":("#f2f2f2", "#444444"),
    "code":   ("#fafafa", "#888888"),
}


# ---------------------------------------------------------------------------
# Primitivas de dibujo
# ---------------------------------------------------------------------------

def _box(ax, x, y, w, h, title, sub=None, ref=None, kind="neutral", fs=11):
    face, edge = C[kind]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
        linewidth=1.6, edgecolor=edge, facecolor=face, mutation_aspect=1))
    cx = x + w / 2
    # Alturas relativas según cuántos elementos hay (evita solapamiento).
    if sub and ref:
        y_title, y_sub, y_ref = y + 0.70 * h, y + 0.42 * h, y + 0.15 * h
    elif sub:
        y_title, y_sub, y_ref = y + 0.62 * h, y + 0.30 * h, None
    elif ref:
        y_title, y_sub, y_ref = y + 0.60 * h, None, y + 0.20 * h
    else:
        y_title, y_sub, y_ref = y + 0.5 * h, None, None
    ax.text(cx, y_title, title, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=edge)
    if sub and y_sub is not None:
        ax.text(cx, y_sub, sub, ha="center", va="center", fontsize=fs - 2.5,
                color="#333333")
    if ref and y_ref is not None:
        ax.text(cx, y_ref, ref, ha="center", va="center", fontsize=fs - 3.5,
                family="monospace", color=C["code"][1])


def _arrow(ax, p0, p1, label=None, style="-|>", color="#555555"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=16, linewidth=1.5,
        color=color, shrinkA=2, shrinkB=2))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + 0.012, my, label, ha="left", va="center", fontsize=8,
                family="monospace", color="#666666")


def _canvas(w=11, h=7):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out.relative_to(OUT_DIR.parents[3])}")


# ---------------------------------------------------------------------------
# A1 — Arquitectura E2E
# ---------------------------------------------------------------------------

def diagram_arquitectura_e2e():
    fig, ax = _canvas(9, 9)
    ax.text(0.5, 0.975, "Arquitectura E2E: fMRI → Adapter → CLIP → SD 2.1 unCLIP",
            ha="center", fontsize=13, fontweight="bold")
    bw, bx = 0.62, 0.19
    _box(ax, bx, 0.80, bw, 0.10, "fMRI — BOLD5000 ROIs",
         "~10k vóxeles visuales (LHLOC, RHLOC, …)", "bold5000_loader.py:207", "in")
    _box(ax, bx, 0.575, bw, 0.145, "Adapter fMRI → CLIP",
         "Ridge · Ridge estocástico · MindEye", "train_adapter.py:124", "adapt")
    _box(ax, bx, 0.40, bw, 0.09, "z_CLIP  ∈  ℝ⁷⁶⁸",
         "embedding CLIP ViT-L/14", "extract_vit_features.py:133", "clip")
    _box(ax, bx, 0.175, bw, 0.145, "SD 2.1 unCLIP",
         "UNet (frozen) + VAE decoder (frozen)", "sd_decoder.py:196", "sd")
    _box(ax, bx, 0.03, bw, 0.09, "Imagen reconstruida",
         "768×768 px", None, "out")
    xc = bx + bw / 2
    _arrow(ax, (xc, 0.80), (xc, 0.722))
    _arrow(ax, (xc, 0.575), (xc, 0.492), "predict")
    _arrow(ax, (xc, 0.40), (xc, 0.322), "image_embeds")
    _arrow(ax, (xc, 0.175), (xc, 0.122))
    _save(fig, "arquitectura_e2e.png")


# ---------------------------------------------------------------------------
# A2 — Flujo de módulos phase2/
# ---------------------------------------------------------------------------

def diagram_flujo_modulos():
    fig, ax = _canvas(13, 6.5)
    ax.text(0.5, 0.96, "Flujo de datos y módulos — src/phase2/",
            ha="center", fontsize=13, fontweight="bold")
    y = 0.58
    w, h, gap = 0.155, 0.16, 0.035
    xs = [0.02 + i * (w + gap) for i in range(5)]
    _box(ax, xs[0], y, w, h, "bold5000_loader\n+ loader.load_split",
         "Split train/test", "loader.py:119", "in", fs=9.5)
    _box(ax, xs[1], y, w, h, "extract_vit_features",
         "targets CLIP ViT-L/14", "clip_targets.pt", "clip", fs=9.5)
    _box(ax, xs[2], y, w, h, "train_adapter",
         "Ridge fit (α=60000)", "embeds_test.pt", "adapt", fs=9.5)
    _box(ax, xs[3], y, w, h, "visual_evaluator",
         "embeds → SD → recon", "reconstructions/", "sd", fs=9.5)
    _box(ax, xs[4], y, w, h, "compare_subjects",
         "[GT|CSI1..CSI4]", "compare_subjects.py:278", "neutral", fs=9.5)
    for i in range(4):
        _arrow(ax, (xs[i] + w, y + h / 2), (xs[i + 1], y + h / 2))
    # fila inferior: build_appendix_montages
    _box(ax, xs[3], 0.14, w + gap + w, h, "build_appendix_montages",
         "galería paginada doble columna", "apendiceB_galeria.{png,tex}", "out", fs=9.5)
    _arrow(ax, (xs[4] + w / 2, y), (xs[4] + w / 2, 0.30))
    ax.text(0.02, 0.05,
            "Cada nodo = script real de src/phase2/. Rutas resueltas por config.py (ACECOM_*).",
            fontsize=8.5, color="#666666", family="monospace")
    _save(fig, "flujo_modulos_phase2.png")


# ---------------------------------------------------------------------------
# A3 — Mapa de módulos del repo
# ---------------------------------------------------------------------------

def diagram_mapa_repo():
    fig, ax = _canvas(11, 8)
    ax.text(0.5, 0.97, "Mapa de módulos del repositorio",
            ha="center", fontsize=13, fontweight="bold")
    _box(ax, 0.02, 0.30, 0.50, 0.60, "src/", None, None, "in", fs=12)
    core = [
        ("config.py", "paths/params (ACECOM_*)"),
        ("sd_decoder.py", "pipeline SD 2.1 unCLIP"),
        ("evaluation.py", "SSIM/LPIPS/CLIP/pairwise"),
        ("extract_metrics.py", "R2/Cosine/MSE"),
        ("locate_recons.py", "resuelve rutas + zip"),
    ]
    for i, (n, d) in enumerate(core):
        yy = 0.80 - i * 0.075
        ax.text(0.05, yy, n, fontsize=9.5, family="monospace", fontweight="bold",
                color=C["in"][1], va="center")
        ax.text(0.24, yy, d, fontsize=8.5, color="#444444", va="center")
    _box(ax, 0.05, 0.32, 0.44, 0.12, "phase2/",
         "loader · adapter_ridge(_stoch) · mindeye_models · train_adapter",
         "extract_vit_features · visual_evaluator · compare_subjects · build_appendix_montages",
         "adapt", fs=10)
    _box(ax, 0.56, 0.72, 0.42, 0.18, "tests/",
         "suite mock, sin GPU (pytest)",
         "loader · ridge_stoch · mindeye · compare · eval", "clip", fs=11)
    _box(ax, 0.56, 0.50, 0.42, 0.14, "exe/",
         "ejecutable.bat — recons CSI1..CSI4", None, "neutral", fs=11)
    _box(ax, 0.56, 0.30, 0.42, 0.14, "AlvaroTaipe_Plantilla/",
         "fuentes LaTeX de la tesis", "capítulos · apéndices · figuras", "out", fs=11)
    _save(fig, "mapa_modulos_repo.png")


# ---------------------------------------------------------------------------
# C1 — Proceso de difusión
# ---------------------------------------------------------------------------

def diagram_proceso_difusion():
    fig, ax = _canvas(12, 6)
    ax.text(0.5, 0.95, "Proceso de difusión — SD 2.1 unCLIP",
            ha="center", fontsize=13, fontweight="bold")
    _box(ax, 0.03, 0.55, 0.16, 0.18, "x₀", "imagen", None, "out", fs=13)
    _box(ax, 0.42, 0.55, 0.16, 0.18, "xₜ", "ruido parcial", None, "neutral", fs=13)
    _box(ax, 0.81, 0.55, 0.16, 0.18, "x_T", r"$\mathcal{N}(0,\,I)$", None, "in", fs=13)
    _arrow(ax, (0.19, 0.64), (0.42, 0.64), None, color="#b5651d")
    _arrow(ax, (0.58, 0.64), (0.81, 0.64), None, color="#b5651d")
    ax.text(0.505, 0.70, "forward  q(xₜ|xₜ₋₁)  añade ruido",
            ha="center", fontsize=9.5, color="#b5651d")
    # reverse
    _arrow(ax, (0.81, 0.42), (0.19, 0.42), None, color="#6b3fb5")
    ax.text(0.50, 0.36, "reverse  εθ(xₜ,t)  — denoising guiado",
            ha="center", fontsize=9.5, color="#6b3fb5")
    _box(ax, 0.30, 0.06, 0.40, 0.16, "Guía CLIP + CFG",
         "z_CLIP (fMRI) · w = 8.0", "sd_decoder.py:196", "clip", fs=10.5)
    _arrow(ax, (0.50, 0.22), (0.50, 0.30), None, color="#6b3fb5")
    ax.text(0.03, 0.30, "x_T ─► x₀ reconstruida", fontsize=9, color="#6b3fb5",
            family="monospace")
    _save(fig, "proceso_difusion.png")


# ---------------------------------------------------------------------------
# C2 — Ridge vs MindEye
# ---------------------------------------------------------------------------

def diagram_ridge_vs_mindeye():
    fig, ax = _canvas(12, 6.5)
    ax.text(0.5, 0.95, "Adaptadores fMRI → CLIP: Ridge · Ridge estocástico · MindEye",
            ha="center", fontsize=13, fontweight="bold")
    w, h, y = 0.30, 0.62, 0.14
    xs = [0.02, 0.35, 0.68]
    _box(ax, xs[0], y, w, h, "Ridge (cerrado)",
         None, "adapter_ridge.py:33", "adapt", fs=12)
    ax.text(xs[0] + w / 2, 0.60, r"$W=(X^\top X+\lambda I)^{-1}X^\top Y$",
            ha="center", fontsize=11)
    for i, t in enumerate(["+ solución exacta", "+ baseline evaluado",
                            "− sesga la norma", "− lineal"]):
        ax.text(xs[0] + 0.02, 0.50 - i * 0.06, t, fontsize=9, color="#444444")
    _box(ax, xs[1], y, w, h, "Ridge estocástico",
         None, "adapter_ridge_stoch.py", "adapt", fs=12)
    ax.text(xs[1] + w / 2, 0.60, "SGD sobre misma pérdida", ha="center", fontsize=9.5)
    for i, t in enumerate(["+ penalizaciones no cerradas", "+ escala a datos grandes",
                           "+ peldaño de contribución", "− requiere tuning LR"]):
        ax.text(xs[1] + 0.02, 0.50 - i * 0.06, t, fontsize=9, color="#444444")
    _box(ax, xs[2], y, w, h, "MindEye",
         None, "mindeye_models.py:84", "sd", fs=12)
    ax.text(xs[2] + w / 2, 0.60, "MLP residual (n_blocks=4)", ha="center", fontsize=9.5)
    for i, t in enumerate(["+ no lineal, profundo", "+ InfoNCE + MSE + Cosine",
                           "+ mayor capacidad", "− caro de entrenar"]):
        ax.text(xs[2] + 0.02, 0.50 - i * 0.06, t, fontsize=9, color="#444444")
    ax.text(0.5, 0.06,
            "Renormalización del embedding (norm_mode 'ridge'/'unit'/'none') "
            "previa a SD 2.1 unCLIP — visual_evaluator.py",
            ha="center", fontsize=8.5, color="#666666", family="monospace")
    _save(fig, "ridge_vs_mindeye.png")


def main() -> int:
    print("Generando diagramas en:", OUT_DIR)
    diagram_arquitectura_e2e()
    diagram_flujo_modulos()
    diagram_mapa_repo()
    diagram_proceso_difusion()
    diagram_ridge_vs_mindeye()
    print("done — 5 diagramas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
