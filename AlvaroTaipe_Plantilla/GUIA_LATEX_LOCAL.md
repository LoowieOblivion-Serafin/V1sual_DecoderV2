# Guía de LaTeX local — Tesis IA-ACECOM

Guía operacional para compilar, depurar y mantener `AlvaroTaipe_Plantilla/` en Windows. Pensada para uso continuo durante la tesis.

---

## 1. Instalación de la distribución

### Opción A — MiKTeX (recomendado en Windows)

1. Descargar instalador desde `https://miktex.org/download` (versión "Basic Installer", 64-bit).
2. Instalar para "All users" o "Only for me" (segundo es suficiente y no pide admin).
3. En la primera ejecución activar "Install missing packages on the fly: Yes" → MiKTeX descarga paquetes que falten al compilar (palatino, tocloft, glossaries, etc.) sin intervención manual.
4. Tras instalar, abrir nueva terminal (cmd, PowerShell o Git Bash) y verificar:

   ```cmd
   pdflatex --version
   ```

   Debe imprimir `pdfTeX 3.x ... MiKTeX ...`.

### Opción B — TeX Live

Distribución completa, ~6 GB. Útil si se trabaja también en Linux/macOS.

1. Descargar `install-tl-windows.exe` desde `https://tug.org/texlive/`.
2. Instalación completa puede tardar 30–60 min.
3. Misma verificación: `pdflatex --version`.

### Editor recomendado

| Editor              | Por qué                                                                  |
|---------------------|---------------------------------------------------------------------------|
| **VS Code + LaTeX Workshop** | Ligero, integra build, autocompletado, vista PDF lado a lado. Recomendado. |
| **TeXstudio**       | Específico LaTeX, panel de errores legible, usa MiKTeX/TeX Live de fondo. |
| **Overleaf** (web)  | Sin instalación; usar para colaborar o backup, no como flujo principal.   |

#### VS Code — extensión LaTeX Workshop

1. VS Code → Extensions → buscar `LaTeX Workshop` (autor `James Yu`) → Install.
2. Abrir `AlvaroTaipe_Plantilla/main.tex`.
3. Sidebar TeX (icono de hoja con TeX) → "Build LaTeX project".
4. Pestaña "View LaTeX PDF" → lado a lado con el `.tex`.

`settings.json` (User o Workspace) recomendado:

```json
{
  "latex-workshop.latex.outDir": "%DIR%",
  "latex-workshop.latex.autoBuild.run": "onSave",
  "latex-workshop.view.pdf.viewer": "tab",
  "latex-workshop.synctex.afterBuild.enabled": true,
  "latex-workshop.latex.recipes": [
    {
      "name": "pdflatex ×2",
      "tools": ["pdflatex", "pdflatex"]
    }
  ],
  "latex-workshop.latex.tools": [
    {
      "name": "pdflatex",
      "command": "pdflatex",
      "args": [
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "%DOC%"
      ]
    }
  ]
}
```

---

## 2. Compilación del proyecto

### Receta canónica (este proyecto)

`main.tex` usa bibliografía manual (`\begin{thebibliography}` ... `\end{thebibliography}`), índices generados por `tocloft` (`\listoffigures`, `\listoftables`, `\listofmyequations`, `\lstlistoflistings`) y referencias cruzadas. Por tanto:

```cmd
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Dos pasadas. Primera escribe `.aux` con etiquetas; segunda resuelve `\ref`, `\pageref`, ToC, LoF, LoT.

Si en algún momento se migra a `biblatex` + `.bib`, la receta cambia:

```cmd
pdflatex main.tex
biber main           :: o "bibtex main" si se usa BibTeX clásico
pdflatex main.tex
pdflatex main.tex
```

### Build script (recomendable)

Guardar como `build.bat` en raíz del proyecto:

```bat
@echo off
setlocal
echo === Pasada 1 ===
pdflatex -interaction=nonstopmode -file-line-error main.tex || goto :err
echo === Pasada 2 ===
pdflatex -interaction=nonstopmode -file-line-error main.tex || goto :err
echo === OK: main.pdf generado ===
exit /b 0
:err
echo === ERROR. Ver main.log ===
exit /b 1
```

Uso: `build.bat` desde la raíz.

Versión PowerShell (`build.ps1`):

```powershell
$ErrorActionPreference = "Stop"
foreach ($i in 1..2) {
  Write-Host "=== Pasada $i ===" -ForegroundColor Cyan
  pdflatex -interaction=nonstopmode -file-line-error main.tex
  if ($LASTEXITCODE -ne 0) { Write-Host "Error. Ver main.log"; exit 1 }
}
Write-Host "OK: main.pdf" -ForegroundColor Green
```

### Limpieza de intermedios

LaTeX deja muchos archivos auxiliares. `clean.bat`:

```bat
@echo off
del /q main.aux main.log main.toc main.lof main.lot main.equ main.lol main.out main.synctex.gz main.bbl main.blg
for %%f in (Chapters\*.aux Appendix\*.aux) do del /q "%%f"
echo Limpiado.
```

No borrar `main.pdf` ni los `.tex`.

---

## 3. Estructura del proyecto

```
AlvaroTaipe_Plantilla/
├── main.tex                         ← raíz, no se compilan los Cap*.tex sueltos
├── MastersDoctoralThesis.cls        ← clase custom (no editar salvo causa fuerte)
├── Chapters/
│   ├── Cap1.tex … Cap9.tex          ← cada uno es \chapter independiente
├── Appendix/
│   └── AppendixA.tex … C.tex
├── Figures/
│   ├── *.png  *.jpg                 ← imágenes
│   └── reconstrucciones/            ← carpetas anidadas OK, ruta relativa desde main.tex
├── main.bib                         ← (opcional, no usado actualmente)
└── main.pdf                         ← salida
```

`main.tex` orquesta vía `\input{Chapters/Cap1}`. Editar capítulos en `Chapters/`; `main.tex` se toca solo para portada, abstract, acrónimos, bibliografía o estructura global.

---

## 4. Patrones LaTeX usados en este proyecto

### 4.1 Capítulos y secciones

```latex
\chapter{Nombre del capítulo}
\section{Sección}
\subsection{Subsección}
\subsubsection{...}
\paragraph{Etiqueta inline.}
```

### 4.2 Ecuaciones numeradas

```latex
\begin{equation}
    \mathcal{L}(\theta) = \sum_i \|\hat{\mathbf{e}}_i - \mathbf{e}_i\|^2.
\end{equation}
\myequations{Pérdida cuadrática}
```

`\myequations{...}` registra entrada en el "Índice de ecuaciones". **Siempre poner una línea** después de `\end{equation}` con título corto, así aparece en el ToC de ecuaciones.

Sistema de varias ecuaciones alineadas:

```latex
\begin{align}
    a &= b + c, \\
    d &= e \cdot f.
\end{align}
```

Ecuación inline: `$x^2 + y^2$`. Ecuación display sin numerar: `\[ ... \]`.

### 4.3 Símbolos comunes

| Concepto              | Comando                              |
|-----------------------|--------------------------------------|
| Esperanza             | `\mathbb{E}`                         |
| Reales                | `\mathbb{R}`                         |
| Naturales             | `\mathbb{N}`                         |
| Norma                 | `\|x\|`, `\|x\|_2`                   |
| Producto interno      | `\langle a, b \rangle`               |
| Conjuntos             | `\mathcal{S}`, `\mathcal{L}`         |
| Negrita matemática    | `\mathbf{x}`                         |
| Hat / Tilde           | `\hat{x}`, `\tilde{x}`               |
| Fracción              | `\frac{a}{b}`                        |
| Sumatoria             | `\sum_{i=1}^{N}`                     |
| Integral              | `\int_a^b f(x)\,dx`                  |
| Implica               | `\Rightarrow`, `\implies`            |
| Aproximadamente       | `\approx`, `\sim`                    |
| Mucho menor           | `\ll`, `\gg`                         |

### 4.4 Figuras

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{Figures/reconstrucciones/desertvegetation3_compare.png}
\caption[Pie corto en LoF]{Pie largo con detalle en cuerpo del capítulo.}
\label{fig:ridge_succ_desert}
\end{figure}
```

Notas:

- `[H]` (paquete `float`) fuerza la figura exactamente donde aparece. Sin paquete `float` el placement por defecto es `[htbp]` y LaTeX puede moverla.
- `\caption[...]{...}` con dos argumentos: el corto va al "Índice de figuras"; el largo, al cuerpo.
- Referenciar luego con `Figura~\ref{fig:ridge_succ_desert}`.
- Rutas con `/`, no `\`, incluso en Windows.
- Extensiones soportadas en pdflatex: `.png`, `.jpg`, `.pdf`. Para `.eps`, convertir a `.pdf` con `epstopdf`.

#### Subfiguras (varias imágenes en una)

```latex
\begin{figure}[H]
\centering
\subfloat[Éxito]{\includegraphics[width=0.45\textwidth]{Figures/reconstrucciones/desertvegetation3_compare.png}}\hfill
\subfloat[Fallo]{\includegraphics[width=0.45\textwidth]{Figures/reconstrucciones/ridge_fail_n01641577_1229_compare.png}}
\caption{Contraste éxito vs.\ fallo del adapter Ridge.}
\label{fig:ridge_contraste}
\end{figure}
```

Paquete `subfig` ya está incluido en `main.tex`.

### 4.5 Tablas

```latex
\begin{table}[H]
\centering
\begin{tabular}{| l | c | c | c |}
\hline
\textbf{Sujeto} & \textbf{Trials} & \textbf{Vóxeles} & \textbf{Sesiones} \\ \hline
CSI1 & 4803 & 5917 & 15 \\ \hline
CSI2 & TBD  & TBD  & 15 \\ \hline
\end{tabular}
\caption[Estadísticas BOLD5000]{Conteo de pares por sujeto.}
\label{table:bold5000_stats}
\end{table}
```

Especificadores de columna: `l` izquierda, `c` centrada, `r` derecha, `p{3cm}` ancho fijo con justificado. Barras `|` dibujan líneas verticales; `\hline` líneas horizontales.

### 4.6 Listados de código

Paquete `listings` ya cargado. Estilo Python:

```latex
\begin{lstlisting}[language=Python, caption=Inferencia por trial, label={lst:infer}, numbers=left]
embed_hat = adapter(fmri_voxels)               # (1, 768)
image = pipeline(
    image_embeds=embed_hat,
    prompt="",
    num_inference_steps=75,
    guidance_scale=8.0,
).images[0]
\end{lstlisting}
```

Bash, R, Matlab, etc.: cambiar `language=`. Para registrar el listing en el "Índice de algoritmos" basta con tener `caption=` y `label=`.

### 4.7 Citas

Bibliografía manual en `main.tex`:

```latex
\bibitem{scotti2023mindeye} P. S. Scotti et al., ``Reconstructing the Mind's Eye'', NeurIPS, 2023.
```

Citar en cuerpo:

```latex
La pérdida InfoNCE bidireccional sigue el patrón de MindEye~\cite{scotti2023mindeye}.
```

Múltiples: `\cite{scotti2023mindeye, scotti2024mindeye2}`.

### 4.8 Itemize y enumerate

```latex
\begin{itemize}
\item[•] Bullet con bullet personalizado.
\item Bullet por defecto.
\end{itemize}

\begin{enumerate}
\item Primero.
\item[(i)] Numeración custom.
\end{enumerate}
```

### 4.9 Espacios y saltos

| Quiero…                         | Comando                                |
|---------------------------------|----------------------------------------|
| Espacio fino                    | `\,`                                   |
| Espacio normal                  | `\ ` (con espacio explícito)           |
| Espacio en blanco vertical      | `\vspace{1em}`                         |
| Salto de página                 | `\newpage`                             |
| Salto de página + clear floats  | `\clearpage`                           |
| Página en blanco                | `\afterpage{\blankpage}` (ya definido) |
| No partir palabra (no break)    | `~` entre palabras (`Cap.~7`, `Fig.~3`) |

### 4.10 Caracteres especiales

| Quiero imprimir | Escribir |
|-----------------|----------|
| `%`             | `\%`     |
| `$`             | `\$`     |
| `&`             | `\&`     |
| `_`             | `\_`     |
| `#`             | `\#`     |
| `{`, `}`        | `\{`, `\}` |
| `~`             | `\textasciitilde{}` |
| `\`             | `\textbackslash{}` |

En modo matemático las reglas cambian (p.\ ej.\ `_` es subíndice).

---

## 5. Errores frecuentes y cómo diagnosticarlos

### 5.1 Lectura de `main.log`

Cuando pdflatex falla, abrir `main.log` y buscar líneas que empiezan con `!`. Ejemplo:

```
! Undefined control sequence.
l.42 \undefinedmacro
                    {arg}
```

`l.42` indica línea del `.tex` original (si se usó `-file-line-error` se muestra `archivo:línea: !`).

### 5.2 Errores típicos

| Síntoma                                       | Causa                                           | Fix                                                       |
|-----------------------------------------------|--------------------------------------------------|------------------------------------------------------------|
| `! Undefined control sequence`                | Comando inexistente o paquete sin cargar         | Verificar ortografía; añadir `\usepackage{...}` en `main.tex` |
| `! Missing $ inserted`                        | Símbolo matemático fuera de `$...$`              | Envolver en `$...$` o `\(...\)`                             |
| `! LaTeX Error: File not found`               | Imagen o archivo `\input{...}` mal nombrado      | Verificar ruta y mayúsculas (Windows tolerante, otros no)  |
| `Overfull \hbox (XX pt too wide)`             | Línea desborda margen                             | Aviso, no error. Si es severo, ajustar texto o usar `\sloppy` |
| `Underfull \hbox`                             | Espaciado feo en línea poco llena                 | Aviso, ignorable                                            |
| `Reference 'XXX' on page Y undefined`         | Falta segunda pasada o etiqueta mal escrita       | Recompilar; verificar `\label{XXX}` existe                  |
| `! Paragraph ended before \xxx was complete`  | Falta `}` o `\end{...}`                           | Buscar bloque sin cerrar                                    |
| `Missing number, treated as zero`             | `\hspace{}` o similar sin unidad o vacío          | Añadir unidad: `\hspace{1cm}`                              |
| Caracteres acentuados se ven mal              | Encoding incorrecto                               | Mantener `\usepackage[utf8]{inputenc}` y guardar `.tex` en UTF-8 |
| `! Package inputenc Error: Unicode character` | Carácter Unicode no traducible                    | Cambiar el carácter o cargar `\usepackage[T1]{fontenc}`     |

### 5.3 Modo de recuperación

Compilación interactiva (sin `-interaction=nonstopmode`) permite ingresar comandos al detener:

- `r` → reanudar (skip error)
- `q` → silencioso, ignora errores subsiguientes
- `x` → abortar
- `h` → ayuda

---

## 6. Buenas prácticas para esta tesis

### 6.1 Etiquetas con prefijo

Convención adoptada en este proyecto:

| Tipo       | Prefijo        | Ejemplo                          |
|------------|----------------|----------------------------------|
| Figura     | `fig:`         | `\label{fig:ridge_succ_desert}`  |
| Tabla      | `table:`       | `\label{table:results_variants}` |
| Ecuación   | `eq:`          | `\label{eq:infonce}`             |
| Listing    | `lst:`         | `\label{lst:infer}`              |
| Sección    | `sec:`         | `\label{sec:metricas}`           |

### 6.2 Comentarios

`%` inicia comentario hasta fin de línea. Comentar bloque entero:

```latex
\begin{comment}
... bloque comentado ...
\end{comment}
```

Requiere `\usepackage{verbatim}` (ya incluido).

### 6.3 Versionado en Git

`.gitignore` recomendado para LaTeX:

```
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.lof
*.log
*.lol
*.lot
*.out
*.synctex.gz
*.toc
*.equ
*.idx
*.ind
*.ilg
*.dvi
```

Mantener versionados `.tex`, `.cls`, `.bib`, `Figures/`, y opcionalmente `main.pdf` si se quiere distribuir compilado.

### 6.4 Ortografía

VS Code → extensión `LTeX – LanguageTool` para revisar gramática y ortografía en español dentro de `.tex` (entiende sintaxis LaTeX y no marca falsos positivos en comandos).

### 6.5 Texto largo en español: tipografía

- `\spacing{1.4}` (interlineado) ya configurado.
- `\decimalpoint` ya configurado (punto decimal en lugar de coma).
- Comilla tipográfica española: `\enquote{texto}` (paquete `csquotes` cargado).
- Énfasis: preferir `\emph{...}` (cursiva contextual) sobre `\textit{...}`.
- Términos técnicos extranjeros: `\emph{...}` la primera vez (`\emph{shrinkage}`), luego sin marcar.

### 6.6 Workflow día a día

1. Abrir `main.tex` en VS Code.
2. Editar el capítulo correspondiente en `Chapters/CapN.tex`.
3. Guardar con `Ctrl+S` → LaTeX Workshop compila si está configurado `onSave`. Si no, `Ctrl+Alt+B`.
4. Revisar `main.pdf` lado a lado.
5. `Ctrl+Alt+J` → SyncTeX salta del PDF al `.tex` y viceversa.
6. Cuando un capítulo queda estable, commit Git.

### 6.7 Trabajar un solo capítulo aislado (tip avanzado)

Compilar todo `main.tex` en cada iteración es lento. Para iterar rápido sobre un capítulo, usar `\includeonly` en el preámbulo de `main.tex`:

```latex
\includeonly{Chapters/Cap7}
```

LaTeX procesa preámbulo, ToC y solo `Cap7`. Quitar antes del build final.

Alternativa: paquete `subfiles`. Más invasivo, requiere reestructuración.

---

## 7. Recursos útiles

| Recurso                          | URL / dónde                                                  |
|----------------------------------|--------------------------------------------------------------|
| Documentación oficial            | `https://www.latex-project.org/help/documentation/`          |
| TeX Stack Exchange               | `https://tex.stackexchange.com/`                             |
| Detexify (símbolo → comando)     | `https://detexify.kirelabs.org/`                             |
| Tablas online (generador)        | `https://www.tablesgenerator.com/`                           |
| Catálogo CTAN de paquetes        | `https://ctan.org/`                                          |
| Lista de errores comunes         | `https://en.wikibooks.org/wiki/LaTeX/Errors_and_Warnings`    |

---

## 8. Comandos custom de este proyecto

Definidos en `main.tex` o en `MastersDoctoralThesis.cls`:

| Comando                  | Uso                                              |
|--------------------------|--------------------------------------------------|
| `\thesistitle{...}`      | Título de la tesis                               |
| `\authorname` / `\author{...}` | Autor                                       |
| `\supname` / `\supervisor{...}` | Asesor                                     |
| `\univname` / `\university{...}` | Universidad                               |
| `\myequations{título}`   | Registra ecuación en "Índice de ecuaciones"      |
| `\blankpage`             | Inserta página en blanco (con `\afterpage{\blankpage}`) |
| `\mainmatter`            | Inicia paginado arábigo del cuerpo               |
| `\frontmatter`           | Inicia paginado romano de las páginas previas    |

---

## 9. Checklist antes de cerrar la tesis

- [ ] Todas las `TBD` resueltas o explícitamente justificadas como "pendiente de corrida X".
- [ ] Todas las `\ref{...}` resuelven (sin "??" en el PDF).
- [ ] Bibliografía completa, sin `\cite{...}` no resueltos.
- [ ] Compila limpio en dos pasadas, sin errores ni warnings críticos.
- [ ] `main.pdf` actualizado y commiteado.
- [ ] Índices (figuras, tablas, ecuaciones, algoritmos) generados correctamente.
- [ ] Acrónimos en `main.tex` cubren todos los usados en cuerpo.
- [ ] Spell check final con LTeX en español.
- [ ] Backup en Overleaf u otro repositorio remoto.
