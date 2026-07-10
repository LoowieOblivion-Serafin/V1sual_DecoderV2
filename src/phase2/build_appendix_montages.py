"""
===============================================================================
phase2/build_appendix_montages.py — Galería paginada para el Apéndice B
===============================================================================

Ensambla las reconstrucciones cross-subject en *tiras largas* listas para
pegar en el apéndice de la tesis. Cada fila es una muestra:

    [ Estímulo original | CSI1 | CSI2 | CSI3 | CSI4 ]

A diferencia de `compare_subjects.py` (que vuelca UNA figura con todas las
filas y no controla el corte de página), este script **pagina**: parte la
lista de estímulos en bloques de `--rows-per-page` filas y emite un PNG por
bloque, todos del mismo ancho. Así se generan las "imágenes largas" que se
insertan como floats a lo ancho de la hoja en el documento a doble columna.

Opcionalmente emite un `.tex` con un `figure*` por página (`--emit-tex`),
listo para `\\input{}` desde `Appendix/AppendixB.tex`.

ENTRADA
-------
Reconstrucciones que escribe `visual_evaluator.py`:

    {eval_dir}/{subject}/reconstructions/{stem}_recon.png

El Ground Truth se resuelve por `stem` recorriendo el árbol de estímulos
BOLD5000 (COCO/ImageNet/Scene), igual que el resto del pipeline.

USO
---
    # Galería completa, 10 filas por página, PNG + snippet LaTeX
    python -m phase2.build_appendix_montages --rows-per-page 10 --emit-tex

    # Subconjunto barajado reproducible (p. ej. 40 muestras)
    python -m phase2.build_appendix_montages --limit 40 --shuffle --seed 7

PORTABILIDAD
------------
Sin rutas hardcodeadas: todo se deriva de `config` y de las variables
`ACECOM_*`. Sólo depende de matplotlib + PIL (sin torch/diffusers), así que
corre en Máquina A (dev) o B (post git-pull, headless) sin GPU.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend headless (Máquina B sin display)
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from phase2.compare_subjects import (
    _load_img,
    _recon_path,
    find_ground_truth,
    select_stems,
)

logger = logging.getLogger("phase2.build_appendix_montages")

DEFAULT_FIG_RELPATH = "Figures/reconstrucciones/apendice"


# ---------------------------------------------------------------------------
# Render de una página (bloque de filas)
# ---------------------------------------------------------------------------

def render_page(
    stems: list[str],
    eval_dir: Path,
    stimuli_root: Path,
    subjects: list[str],
    out_path: Path,
    cell_px: int,
    dpi: int,
    show_header: bool,
) -> int:
    """
    Dibuja un bloque: len(stems) filas × (1 + len(subjects)) columnas.
    Devuelve el número de celdas de reconstrucción ausentes (placeholder).
    """
    n_rows = len(stems)
    n_cols = 1 + len(subjects)
    col_titles = ["Estímulo"] + list(subjects)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.2 * n_cols, 2.2 * n_rows),
        squeeze=False,
    )

    n_missing = 0
    for i, stem in enumerate(stems):
        gt_path = find_ground_truth(stimuli_root, stem)
        cells = [gt_path] + [_recon_path(eval_dir, s, stem) for s in subjects]
        for j, cell_path in enumerate(cells):
            ax = axes[i][j]
            if j > 0 and (cell_path is None or not cell_path.is_file()):
                n_missing += 1
            ax.imshow(_load_img(cell_path, cell_px))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0 and show_header:
                ax.set_title(col_titles[j], fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(stem, fontsize=7, rotation=0, ha="right",
                              va="center", labelpad=28)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return n_missing


# ---------------------------------------------------------------------------
# Emisión del snippet LaTeX
# ---------------------------------------------------------------------------

def write_tex_snippet(
    tex_path: Path,
    page_names: list[str],
    fig_relpath: str,
    subjects: list[str],
    span: str,
    label_prefix: str,
) -> None:
    """
    Escribe un `.tex` con un float por página, para `\\input` desde el apéndice.
    span='page' usa figure* (ancho de página en doble columna); span='column'
    usa figure normal (ancho de una columna).
    """
    env = "figure*" if span == "page" else "figure"
    width = r"\textwidth" if span == "page" else r"\linewidth"
    n = len(page_names)
    subj_txt = ", ".join(subjects) if subjects else ""

    lines: list[str] = [
        "% Auto-generado por phase2/build_appendix_montages.py — no editar a mano.",
        "% Regenerar tras nuevas reconstrucciones y volver a compilar.",
        "",
    ]
    for k, name in enumerate(page_names, 1):
        label = f"{label_prefix}:p{k:02d}"
        lines += [
            f"\\begin{{{env}}}[htbp]",
            "  \\centering",
            f"  \\includegraphics[width={width}]{{{fig_relpath}/{name}}}",
            f"  \\caption[Galería de reconstrucciones ({k}/{n})]{{Reconstrucciones "
            f"cross-subject sobre estímulos BOLD5000 (parte {k} de {n}). "
            f"Columna~1: estímulo original; columnas~2--{1 + len(subjects)}: "
            f"reconstrucción SD~2.1 unCLIP por sujeto ({subj_txt}).}}",
            f"  \\label{{fig:{label}}}",
            f"\\end{{{env}}}",
            "\\clearpage",
            "",
        ]

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("snippet LaTeX -> %s (%d floats)", tex_path, n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _chunk(seq: list[str], size: int) -> list[list[str]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Genera la galería paginada [GT|CSI1..CSI4] para el Apéndice B."
    )
    ap.add_argument("--subjects", nargs="+", default=list(config.BOLD5000_SUBJECTS),
                    choices=list(config.BOLD5000_SUBJECTS),
                    help="Sujetos por fila (default: los 4).")
    ap.add_argument("--eval-dir", type=Path, default=None,
                    help="Raíz de reconstrucciones. Default: config.DATA_DIRS['eval_output'].")
    ap.add_argument("--stimuli-root", type=Path, default=None,
                    help="Raíz de estímulos GT. Default: config.BOLD5000_CONFIG['stimuli_images'].")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Dir de salida de los PNG paginados. "
                         "Default: {repo}/AlvaroTaipe_Plantilla/" + DEFAULT_FIG_RELPATH)
    ap.add_argument("--prefix", default="apendiceB_galeria",
                    help="Prefijo de los PNG de página.")
    ap.add_argument("--rows-per-page", type=int, default=10,
                    help="Filas (estímulos) por página. Controla el alto de la tira.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap total de estímulos (debug o galería parcial).")
    ap.add_argument("--stem", nargs="+", default=None,
                    help="Stems explícitos (ignora --limit/--shuffle).")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cell-px", type=int, default=256, help="Resolución por celda.")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--emit-tex", action="store_true",
                    help="Escribe también un snippet .tex con un float por página.")
    ap.add_argument("--span", choices=["page", "column"], default="column",
                    help="figure a ancho de columna (default, plantilla single-column) "
                         "o figure* a ancho de página (documento a doble columna).")
    ap.add_argument("--fig-relpath", default=DEFAULT_FIG_RELPATH,
                    help="Ruta que usará \\includegraphics dentro del .tex.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    eval_dir = Path(args.eval_dir or config.DATA_DIRS["eval_output"])
    stimuli_root = Path(args.stimuli_root or config.BOLD5000_CONFIG["stimuli_images"])
    out_dir = Path(
        args.out_dir
        or (config.PROJECT_ROOT / "AlvaroTaipe_Plantilla" / args.fig_relpath)
    )

    logger.info("subjects      = %s", args.subjects)
    logger.info("eval_dir      = %s", eval_dir)
    logger.info("stimuli_root  = %s", stimuli_root)
    logger.info("out_dir       = %s", out_dir)
    logger.info("rows_per_page = %d", args.rows_per_page)

    stems = select_stems(
        eval_dir=eval_dir,
        subjects=args.subjects,
        explicit=args.stem,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not stems:
        logger.error(
            "Sin estímulos comunes a %s en %s. Corre `visual_evaluator` "
            "para cada sujeto antes de ensamblar la galería.",
            args.subjects, eval_dir,
        )
        return 1

    pages = _chunk(stems, max(1, args.rows_per_page))
    page_names: list[str] = []
    total_missing = 0
    for k, page_stems in enumerate(pages, 1):
        name = f"{args.prefix}_p{k:02d}.png"
        out_path = out_dir / name
        missing = render_page(
            stems=page_stems,
            eval_dir=eval_dir,
            stimuli_root=stimuli_root,
            subjects=args.subjects,
            out_path=out_path,
            cell_px=args.cell_px,
            dpi=args.dpi,
            show_header=True,
        )
        total_missing += missing
        page_names.append(name)
        logger.info("página %d/%d -> %s (%d filas, %d celdas ausentes)",
                    k, len(pages), name, len(page_stems), missing)

    if args.emit_tex:
        write_tex_snippet(
            tex_path=out_dir / f"{args.prefix}.tex",
            page_names=page_names,
            fig_relpath=args.fig_relpath,
            subjects=args.subjects,
            span=args.span,
            label_prefix=args.prefix,
        )

    logger.info(
        "done — %d estímulos, %d páginas, %d celdas ausentes -> %s",
        len(stems), len(pages), total_missing, out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
