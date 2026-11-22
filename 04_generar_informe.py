"""Paso 4 — Genera el informe final en HTML, PDF (Edge headless) y DOCX (Microsoft Word).

Entrada: docs/informe.md (Markdown con tablas; las imágenes usan rutas relativas a la raíz del proyecto).
Salida:  docs/Informe_Actividad6_PeritIA.html / .pdf / .docx

Uso:
    python 04_generar_informe.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import markdown

from src import config

DOCS = config.ROOT / "docs"
SRC = DOCS / "informe.md"
NAME = "Informe_Actividad6_PeritIA"
AUTHOR = "Jherson Guzmán"
TITLE = "PeritIA — Clasificación de daños en vehículos (Actividad 6)"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

CSS = """
@page { size: A4; margin: 2.2cm 2cm; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
       color: #1d1d1f; max-width: 820px; margin: 0 auto; }
h1 { font-size: 20pt; color: #0b3d91; margin-bottom: .2em; }
h2 { font-size: 14pt; color: #0b3d91; border-bottom: 2px solid #d0dbf0; padding-bottom: .15em;
     margin-top: 1.4em; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #23395d; margin-top: 1.1em; page-break-after: avoid; }
table { border-collapse: collapse; margin: .6em 0 1em; font-size: 9pt; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #c8cfdb; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #eef3fb; }
code { font-family: Consolas, monospace; font-size: 8.8pt; background: #f3f4f6; padding: 0 3px; }
pre { background: #f3f4f6; padding: 8px; font-size: 8.3pt; white-space: pre-wrap; page-break-inside: avoid; }
img.wide { max-width: 100%; max-height: none; }
img.screen { max-width: 92%; max-height: 17cm; border: 1px solid #c8cfdb; }
img { max-width: 72%; max-height: 8.5cm; display: block; margin: .5em auto .2em; page-break-inside: avoid; }
p.caption, em.caption { display: block; text-align: center; font-size: 9pt; color: #555; margin-bottom: 1em; }
blockquote { border-left: 4px solid #d0dbf0; margin: .6em 0; padding: .2em .8em; color: #444; }
.portada { text-align: center; padding-top: 5cm; page-break-after: always; }
.portada h1 { font-size: 24pt; }
.portada p { font-size: 12pt; margin: .35em 0; }
"""


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
    import re
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
    md_text = SRC.read_text(encoding="utf-8")
    # Python-Markdown exige sangría de 4 espacios para listas anidadas: normaliza 2-3 espacios -> 4
    fixed = []
    for line in md_text.splitlines():
        k = len(line) - len(line.lstrip(" "))
        if 0 < k and line.strip():
            line = " " * (-(-k // 3) * 4) + line.lstrip(" ")
        fixed.append(line)
    md_text = "\n".join(fixed)
    body =markdown.markdown(md_text, extensions=["tables", "fenced_code", "md_in_html", "attr_list"])
    # rutas de imágenes relativas a docs/
    body = body.replace('src="resultados/', 'src="../resultados/')
    html = (f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>PeritIA — Informe Actividad 6</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    html_path = DOCS / f"{NAME}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML  -> {html_path}")

    pdf_path = DOCS / f"{NAME}.pdf"
    subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()], check=True, timeout=120)
    set_pdf_metadata(pdf_path)
    print(f"PDF   -> {pdf_path}")

    docx_path = DOCS / f"{NAME}.docx"
    ps = (
        "$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
        f"$d = $w.Documents.Open('{html_path}'); "
        # incrusta las imágenes en el .docx en vez de dejarlas enlazadas
        "foreach ($s in $d.InlineShapes) { if ($s.LinkFormat) { $s.LinkFormat.SavePictureWithDocument = $true; "
        "$s.LinkFormat.BreakLink() } }; "
        f"$d.SaveAs2('{docx_path}', 16); $d.Close(); $w.Quit()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, timeout=180)
    set_docx_metadata(docx_path)
    print(f"DOCX  -> {docx_path}")


if __name__ == "__main__":
    main()
