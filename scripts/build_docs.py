"""Render repository Markdown docs into a simple, readable HTML page.

Usage:
    python scripts/build_docs.py
    python scripts/build_docs.py docs/architecture.md docs/related-work.md
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify_unicode

ROOT = Path(__file__).resolve().parents[1]

PAGE_TITLES = {
    "architecture.md": "LabScribe 原理与链路详解",
    "related-work.md": "相关方案调研",
    "README.md": "LabScribe",
}

TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
<main>
{body}
<footer>由 <code>scripts/build_docs.py</code> 从 <code>{source}</code> 生成</footer>
</main>
</body>
</html>
"""

CSS = """
:root { color-scheme: light; }
body {
  margin: 0;
  padding: 32px 20px 64px;
  background: #fff;
  color: #222;
  font: 16px/1.75 -apple-system, "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
}
main { max-width: 860px; margin: 0 auto; }
h1, h2, h3, h4 { line-height: 1.35; color: #111; }
h1 { font-size: 28px; margin: 0 0 20px; }
h2 { font-size: 21px; margin: 36px 0 12px; padding-bottom: 6px; border-bottom: 1px solid #e5e5e5; }
h3 { font-size: 17px; margin: 24px 0 8px; }
h4 { font-size: 15px; margin: 18px 0 6px; color: #555; }
p { margin: 10px 0; }
a { color: #0b5cad; text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border: 0; border-top: 1px solid #e5e5e5; margin: 28px 0; }
ul, ol { padding-left: 24px; }
li { margin: 3px 0; }

blockquote {
  margin: 14px 0;
  padding: 10px 16px;
  background: #f6f8fa;
  border-left: 3px solid #888;
  color: #333;
}
blockquote p { margin: 0; }

table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 14px; }
th, td { border: 1px solid #ddd; padding: 7px 10px; text-align: left; vertical-align: top; }
th { background: #f6f8fa; }

code {
  font-family: Consolas, "Cascadia Code", Menlo, monospace;
  font-size: 0.88em;
  background: #f3f4f6;
  padding: 2px 5px;
  border-radius: 4px;
}
pre {
  margin: 14px 0;
  padding: 14px 16px;
  background: #f6f8fa;
  border: 1px solid #e5e5e5;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 13px;
  line-height: 1.6;
}
pre code { background: none; padding: 0; font-size: 13px; }

img { max-width: 100%; display: block; margin: 14px auto; border: 1px solid #e5e5e5; border-radius: 4px; }

footer {
  max-width: 860px;
  margin: 40px auto 0;
  padding-top: 12px;
  border-top: 1px solid #e5e5e5;
  color: #888;
  font-size: 13px;
}

@media print {
  body { padding: 0; }
  pre { white-space: pre-wrap; }
}
"""


def render(source: Path) -> Path:
    text = source.read_text(encoding="utf-8")
    md = markdown.Markdown(
        extensions=["fenced_code", "tables", "toc", "sane_lists", "attr_list"],
        extension_configs={
            "toc": {"toc_depth": "2-3", "permalink": False, "slugify": slugify_unicode}
        },
    )
    body = md.convert(text)
    title = PAGE_TITLES.get(source.name)
    if not title:
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
    title = title or source.stem
    page = TEMPLATE.format(
        title=html.escape(title),
        css=CSS,
        body=body,
        source=html.escape(source.name),
    )
    target = source.with_suffix(".html")
    target.write_text(page, encoding="utf-8")
    return target


def main(argv: list[str]) -> int:
    if argv:
        sources = [Path(a) if Path(a).is_absolute() else ROOT / a for a in argv]
    else:
        sources = [ROOT / "docs" / "architecture.md", ROOT / "docs" / "related-work.md"]
    for source in sources:
        if not source.exists():
            print(f"skip (missing): {source}", file=sys.stderr)
            continue
        target = render(source)
        print(
            f"{source.relative_to(ROOT)} -> {target.relative_to(ROOT)} "
            f"({target.stat().st_size:,} bytes)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
