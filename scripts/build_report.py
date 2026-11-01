#!/usr/bin/env python
"""docs/REPORT.md -> outputs/report/説明資料_PAI最終課題_51550.pdf via Chrome's print-to-PDF.

    uv run --with markdown --with pypdf python scripts/build_report.py [out.pdf]

Markdown -> HTML (tables, fenced code) with a print stylesheet (A4, Hiragino), then headless Chrome prints it.
The PDF gets no author metadata, and the script fails if a name, affiliation or e-mail pattern is in the text.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import markdown
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "REPORT.md"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "report" / "説明資料_PAI最終課題_51550.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
# Generic patterns here; personal ones (name, e-mail) live in the gitignored .personal_patterns (one regex per
# line) so that this script itself carries no personal information into the public repository.
_extra = [l.strip() for l in (ROOT / ".personal_patterns").read_text().splitlines() if l.strip()] \
    if (ROOT / ".personal_patterns").exists() else []
FORBIDDEN = re.compile("|".join([r"@gmail", r"/Users/", *_extra]), re.I)

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Hiragino Sans", "Hiragino Kaku Gothic ProN", sans-serif; font-size: 10.5pt; line-height: 1.6;
       color: #222; max-width: 100%; }
h1 { font-size: 17pt; line-height: 1.4; margin: 0 0 4pt; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt; border-bottom: 1.5px solid #2a9d8f; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
h4 { font-size: 10.5pt; margin: 8pt 0 2pt; }
p { margin: 4pt 0; }
ul, ol { margin: 2pt 0 4pt; padding-left: 1.6em; }
li { margin: 1pt 0; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 9.5pt; }
th, td { border: 1px solid #bbb; padding: 2pt 6pt; }
th { background: #eef5f4; white-space: nowrap; }
td:not(:first-child) { white-space: nowrap; }
img { max-width: 100%; display: block; margin: 6pt auto; }
img[src$="flywheel.png"], img[src$="flywheel_modes.png"], img[src$="audit.png"], img[src$="deploy_flywheel.png"] { max-width: 78%; }
pre { font-family: Menlo, monospace; font-size: 8.5pt; line-height: 1.35; background: #f6f6f6; padding: 6pt;
      white-space: pre-wrap; word-break: break-all; }
code { font-family: Menlo, monospace; font-size: 9pt; }
h2, h3, h4, table, img, pre { page-break-inside: avoid; }
h2, h3, h4 { page-break-after: avoid; }
"""


def main() -> None:
    md = SRC.read_text()
    if m := FORBIDDEN.search(md):
        sys.exit(f"refusing: personal info pattern {m.group(0)!r} in {SRC}")
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    html = (f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><base href="{SRC.parent.as_uri()}/">'
            f"<title>説明資料_PAI最終課題_51550</title><style>{CSS}</style></head><body>{body}</body></html>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    page = OUT.parent / "report.html"
    page.write_text(html)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
                    f"--print-to-pdf={OUT}", page.as_uri()], check=True, capture_output=True,
                   env={**os.environ, "HOME": os.environ.get("HOME", "/tmp")})
    r = PdfReader(str(OUT))
    text = "\n".join(p.extract_text() or "" for p in r.pages)
    meta = {k: v for k, v in (r.metadata or {}).items()}
    if m := FORBIDDEN.search(text + " ".join(map(str, meta.values()))):
        sys.exit(f"refusing: personal info pattern {m.group(0)!r} in the PDF")
    print(f"wrote {OUT} ({len(r.pages)} pages, {OUT.stat().st_size / 1e6:.1f} MB); metadata: {meta}")


if __name__ == "__main__":
    main()
