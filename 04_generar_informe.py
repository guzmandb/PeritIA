"""Paso 4 — Genera el informe final en PDF (Typst) y DOCX (Pandoc).

Entrada: docs/informe.md (Markdown; las imágenes usan rutas relativas a la raíz del proyecto y cada
         figura va seguida de un <p class="caption">Figura N. ...</p>).
Salida:  docs/Informe_Actividad6_PeritIA.pdf / .docx

El PDF reproduce el formato de los informes de la maestría: portada con el logo de la Universidad
de La Salle dentro de un marco, fuente Computer Modern, A4 con márgenes de 2 cm y tablas con filetes.
Requiere Quarto (incluye Pandoc y Typst).

Uso:
    python 04_generar_informe.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from src import config

ROOT = config.ROOT
DOCS = ROOT / "docs"
SRC = DOCS / "informe.md"
NAME = "Informe_Actividad6_PeritIA"
QUARTO = "quarto"

AUTHOR = "Jherson Guzman Ramirez"
PROJECT = "PeritIA: clasificación automática de daños en vehículos"
TITLE = f"{PROJECT} — Proyecto Final"
SUBTITLE = "Proyecto Final – Implementemos un modelo de clasificación"
COURSE = "Visión por Computador G1"
TEACHER = "Carlos Alejandro Espejo Villarraga"
DATE = "22 de noviembre de 2026"

WIDTHS = {"wide": "100%", "screen": "80%", None: "66%"}

PREAMBLE = r"""
#set document(title: "__TITLE__", author: "__AUTHOR__")
#set text(font: "New Computer Modern", size: 10pt, lang: "es")
#set page(paper: "a4", margin: 2cm, numbering: "1", number-align: center)
#set par(justify: true, leading: 0.58em, spacing: 0.9em)
#set heading(numbering: none)
#show heading: set block(above: 1.4em, below: 0.8em)
#show heading.where(level: 1): set text(size: 14.3pt, weight: "bold")
#show heading.where(level: 2): set text(size: 12pt, weight: "bold")
#show heading.where(level: 3): set text(size: 10.5pt, weight: "bold")
#show link: set text(fill: rgb("#1f3fbf"))
#show raw: set text(font: "DejaVu Sans Mono", size: 7.6pt)
#show raw.where(block: true): block.with(fill: luma(246), inset: 7pt, radius: 2pt, width: 100%)
#set table(stroke: (x, y) => if y == 0 { (top: 0.8pt, bottom: 0.45pt) }, inset: (x: 5pt, y: 3.2pt))
#show table: t => block(stroke: (bottom: 0.8pt))[#t]
#show table: set text(size: 8.6pt)
#show table.cell.where(y: 0): set text(weight: "bold")
#show figure: set block(breakable: false, above: 1.1em, below: 1.1em)
#show figure.where(kind: image): set figure(supplement: [Figura], placement: auto)
#show figure.caption: set text(size: 9.3pt)
#set figure.caption(separator: [: ])
#set list(indent: 0.6em)

#page(numbering: none, margin: 2cm,
  background: place(center + horizon, rect(width: 100% - 2cm, height: 100% - 2cm, stroke: 0.7pt)))[
  #set align(center)
  #set par(justify: false)
  #v(0.6cm)
  #image("plantilla/logo_lasalle.png", width: 8cm)
  #v(0.25cm)
  #text(size: 17.2pt)[Universidad de La Salle] \
  #v(0.05cm)
  #text(size: 17.2pt)[Maestría en Inteligencia Artificial]
  #v(0.9cm)
  #line(length: 85%, stroke: 0.9pt)
  #v(0.35cm)
  #block(width: 80%)[#text(size: 24.8pt, weight: "bold")[__PROJECT__]]
  #v(0.35cm)
  #line(length: 85%, stroke: 0.9pt)
  #v(1.8cm)
  #text(size: 17.2pt, weight: "bold")[__SUBTITLE__]
  #v(1.6cm)
  #text(size: 17.2pt, style: "italic")[Autor:]
  #v(0.15cm)
  #text(size: 14.3pt)[__AUTHOR__]
  #v(2.6cm)
  #text(size: 17.2pt)[__COURSE__] \
  #v(0.05cm)
  #text(size: 17.2pt)[Docente: __TEACHER__]
  #v(1.3cm)
  #text(size: 12pt)[__DATE__]
]
#counter(page).update(1)

"""


def html_inline_to_md(s: str) -> str:
    s = re.sub(r"<code>(.*?)</code>", r"`\1`", s)
    s = re.sub(r"<(?:i|em)>(.*?)</(?:i|em)>", r"*\1*", s)
    s = re.sub(r"<(?:b|strong)>(.*?)</(?:b|strong)>", r"**\1**", s)
    return s


def fixed_figures(md: str, window: int = 1500) -> set[int]:
    """Índices de las figuras que van justo después de un título de sección (##): esas no flotan,
    para que no suban por encima del título."""
    fixed = set()
    figs = [m.start() for m in re.finditer(r"^!\[", md, flags=re.M)]
    h2 = [m.start() for m in re.finditer(r"^## ", md, flags=re.M)]
    for i, pos in enumerate(figs):
        prev_h2 = max((h for h in h2 if h < pos), default=-10**9)
        prev_fig = figs[i - 1] if i else -10**9
        if prev_h2 > prev_fig and pos - prev_h2 < window:
            fixed.add(i)
    return fixed


def to_pandoc_md(md: str, img_prefix: str) -> str:
    """Convierte las figuras (imagen + <p class="caption">) en figuras nativas de Pandoc."""
    pat = re.compile(r'!\[[^\]]*\]\(([^)]+)\)(?:\{:\s*\.(\w+)\})?\s*\n<p class="caption">Figura \d+\.\s*(.*?)</p>')

    def rep(m):
        path, cls, cap = m.group(1), m.group(2), html_inline_to_md(m.group(3))
        return f"![{cap}]({img_prefix}{path}){{width={WIDTHS.get(cls, WIDTHS[None])}}}"

    return pat.sub(rep, md)


def pin_figures(typ: str, fixed: set[int]) -> str:
    """Quita la flotación a las figuras indicadas (en orden de aparición)."""
    parts = typ.split("#figure(image(")
    out = parts[0]
    for i, part in enumerate(parts[1:]):
        out += ("#figure(placement: none, image(" if i in fixed else "#figure(image(") + part
    return out


def pandoc(args: list[str], text: str) -> str:
    r = subprocess.run([QUARTO, "pandoc", *args], input=text.encode("utf-8"), capture_output=True, check=True)
    return r.stdout.decode("utf-8").replace("\r\n", "\n")


def tidy_tables(typ: str) -> str:
    """Tablas al estilo LaTeX: columnas automáticas, texto alineado a la izquierda y sin filete duplicado."""
    typ = re.sub(r"\n\s*align: \([^)]*\),", "", typ)
    typ = re.sub(r"\n\s*table\.hline\(\),", "", typ)
    return typ


def set_pdf_metadata(pdf_path: Path):
    """Deja en el PDF el autor y el título del informe."""
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=PdfReader(str(pdf_path)))
    writer.metadata = {"/Title": TITLE, "/Author": AUTHOR, "/Subject": "Visión por Computador — Universidad de La Salle",
                       "/Creator": "Informe académico", "/Producer": "PDF"}
    tmp = pdf_path.with_suffix(".tmp.pdf")
    with open(tmp, "wb") as f:
        writer.write(f)
    tmp.replace(pdf_path)


def set_docx_metadata(docx_path: Path):
    """Escribe autor y título en docProps/core.xml del .docx."""
    import zipfile
    from xml.sax.saxutils import escape

    tmp = docx_path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(docx_path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/core.xml":
                xml = data.decode("utf-8")
                for tag, val in (("dc:creator", AUTHOR), ("cp:lastModifiedBy", AUTHOR), ("dc:title", TITLE)):
                    if f"<{tag}" in xml:
                        xml = re.sub(rf"<{tag}>.*?</{tag}>|<{tag}/>", f"<{tag}>{escape(val)}</{tag}>", xml, flags=re.S)
                    else:
                        xml = xml.replace("</cp:coreProperties>", f"<{tag}>{escape(val)}</{tag}></cp:coreProperties>")
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(docx_path)


def main():
    md = SRC.read_text(encoding="utf-8")
    common = ["-f", "markdown", "--shift-heading-level-by=-1", "--wrap=none", "--columns=10000"]

    # ---- PDF (Typst) ----
    body = tidy_tables(pandoc([*common, "-t", "typst"], to_pandoc_md(md, "../")))
    body = pin_figures(body, fixed_figures(md))
    pre = PREAMBLE
    for k, v in {"TITLE": TITLE, "AUTHOR": AUTHOR, "PROJECT": PROJECT, "SUBTITLE": SUBTITLE,
                 "COURSE": COURSE, "TEACHER": TEACHER, "DATE": DATE}.items():
        pre = pre.replace(f"__{k}__", v)
    typ = DOCS / f"{NAME}.typ"
    typ.write_text(pre + body, encoding="utf-8", newline="\n")
    pdf_path = DOCS / f"{NAME}.pdf"
    subprocess.run([QUARTO, "typst", "compile", str(typ), str(pdf_path), "--root", str(ROOT)], check=True)
    typ.unlink()
    set_pdf_metadata(pdf_path)
    print(f"PDF   -> {pdf_path}")

    # ---- DOCX (Pandoc) ----
    cover = (f"::: {{custom-style=\"Title\"}}\n{PROJECT}\n:::\n\n"
             f"Universidad de La Salle — Maestría en Inteligencia Artificial\n\n{SUBTITLE}\n\n"
             f"Autor: {AUTHOR}\n\n{COURSE} — Docente: {TEACHER}\n\n{DATE}\n\n")
    docx_path = DOCS / f"{NAME}.docx"
    subprocess.run([QUARTO, "pandoc", *common, "-o", str(docx_path), "--resource-path", str(ROOT)],
                   input=(cover + to_pandoc_md(md, "")).encode("utf-8"), check=True)
    set_docx_metadata(docx_path)
    print(f"DOCX  -> {docx_path}")


if __name__ == "__main__":
    main()
