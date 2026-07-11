"""
===============================================================================
locate_recons.py — ¿Dónde quedaron las reconstrucciones tras la última corrida?
===============================================================================

Imprime, resueltas desde `config` (respetando las variables ACECOM_*), las
rutas reales donde `visual_evaluator.py` escribió las reconstrucciones y donde
`build_appendix_montages.py` dejó la galería del apéndice. Cuenta archivos y
tamaño por sujeto, y emite un comando `Compress-Archive` listo para empaquetar
solo lo mínimo necesario para regenerar la galería en otra máquina.

Uso (desde src/):
    python locate_recons.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

RECON_SUFFIX = "_recon.png"


def _dir_stats(d: Path, pattern: str) -> tuple[int, int]:
    """(n_archivos, bytes_totales) para `pattern` dentro de `d`."""
    if not d.is_dir():
        return 0, 0
    files = list(d.glob(pattern))
    return len(files), sum(f.stat().st_size for f in files)


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024 or unit == "GB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} GB"


def main() -> int:
    eval_dir = Path(config.DATA_DIRS["eval_output"])
    stimuli = Path(config.BOLD5000_CONFIG["stimuli_images"])
    apendice = config.PROJECT_ROOT / "AlvaroTaipe_Plantilla" / "Figures" / "reconstrucciones" / "apendice"

    print("=" * 72)
    print("UBICACIÓN DE RECONSTRUCCIONES (resuelto desde config + ACECOM_*)")
    print("=" * 72)
    print(f"PROJECT_ROOT      : {config.PROJECT_ROOT}")
    print(f"eval_output       : {eval_dir}   {'[OK]' if eval_dir.is_dir() else '[NO EXISTE]'}")
    print(f"stimuli_images    : {stimuli}   {'[OK]' if stimuli.is_dir() else '[NO EXISTE]'}")
    print(f"galería apéndice  : {apendice}   {'[OK]' if apendice.is_dir() else '[pendiente]'}")
    print("-" * 72)

    grand_n = grand_b = 0
    for subj in config.BOLD5000_SUBJECTS:
        rdir = eval_dir / subj / "reconstructions"
        n, b = _dir_stats(rdir, f"*{RECON_SUFFIX}")
        grand_n += n
        grand_b += b
        flag = "" if n else "   <-- sin reconstrucciones"
        print(f"  {subj}: {n:4d} recon PNG  ({_human(b):>9}){flag}")
    print("-" * 72)
    print(f"TOTAL recon PNG   : {grand_n} archivos, {_human(grand_b)}")

    ga_n, ga_b = _dir_stats(apendice, "*.png")
    print(f"Galería PNG       : {ga_n} archivos, {_human(ga_b)}")
    print("=" * 72)

    # Comando de empaquetado. Se comprime la CARPETA completa (no `\*`) para
    # conservar el árbol CSIx/reconstructions/...; si se usara un comodín que
    # matchea varias `reconstructions`, colisionarían al mismo nombre en la raíz
    # del zip. La otra máquina ya tiene los estímulos, así que con esto basta
    # para regenerar la galería.
    print("\nPara empaquetar las reconstrucciones (conserva el árbol CSIx/):\n")
    print("  # PowerShell:")
    print(f'  Compress-Archive -Path "{eval_dir}" '
          '-DestinationPath "$env:USERPROFILE\\Desktop\\recons_4070.zip" -Force')
    print("\n  En la otra máquina: extraer el zip en la RAÍZ del repo, de modo que")
    print(f"  quede {eval_dir.name}/CSIx/reconstructions/...  (fusiona con lo existente).")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
