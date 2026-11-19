"""Render repository Markdown docs into a dark, syntax-highlighted HTML page.

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
from pygments.formatters import HtmlFormatter

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
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><rect width='64' height='64' rx='14' fill='%230d1117'/><text x='32' y='44' font-size='28' font-family='Arial' font-weight='bold' text-anchor='middle' fill='%2337d0b0'>LS</text></svg>" />
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
:root { color-scheme: dark; }
body {
  margin: 0;
  padding: 32px 20px 64px;
  background: #0d1117;
  color: #c9d5e2;
  font: 16px/1.78 -apple-system, "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
}
main { max-width: 880px; margin: 0 auto; }
h1, h2, h3, h4 { line-height: 1.35; color: #eef4fb; }
h1 { font-size: 29px; margin: 0 0 20px; }
h2 { font-size: 21px; margin: 38px 0 12px; padding-bottom: 6px; border-bottom: 1px solid #24303d; }
h3 { font-size: 17px; margin: 26px 0 8px; }
h4 { font-size: 15px; margin: 18px 0 6px; color: #9fb0c3; }
p { margin: 10px 0; }
a { color: #6cb6ff; text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border: 0; border-top: 1px solid #24303d; margin: 30px 0; }
ul, ol { padding-left: 24px; }
li { margin: 3px 0; }
strong { color: #f2f7fd; }

blockquote {
  margin: 14px 0;
  padding: 10px 16px;
  background: #121b24;
  border-left: 3px solid #37d0b0;
  color: #bfd4e4;
}
blockquote p { margin: 0; }

table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 14px; }
th, td { border: 1px solid #24303d; padding: 7px 10px; text-align: left; vertical-align: top; }
th { background: #161f29; color: #eaf2fa; }
tr:nth-child(even) td { background: #101720; }

code {
  font-family: Consolas, "Cascadia Code", Menlo, monospace;
  font-size: 0.88em;
  background: #18222d;
  color: #a9e6d4;
  padding: 2px 5px;
  border-radius: 4px;
}
pre {
  margin: 14px 0;
  padding: 14px 16px;
  background: #0a0f15;
  border: 1px solid #24303d;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 13px;
  line-height: 1.62;
}
pre code { background: none; padding: 0; color: inherit; font-size: 13px; }

img { max-width: 100%; display: block; margin: 16px auto; border: 1px solid #24303d; border-radius: 4px; }

footer {
  max-width: 880px;
  margin: 42px auto 0;
  padding-top: 12px;
  border-top: 1px solid #24303d;
  color: #7b8b9d;
  font-size: 13px;
}

@media print {
  body { background: #fff; color: #111; padding: 0; }
  h1, h2, h3, h4, strong { color: #000; }
  pre, code, table, blockquote { background: #f6f8fa; color: #111; border-color: #ddd; }
  a { color: #0b5cad; }
}
"""


def codehilite_css() -> str:
    formatter = HtmlFormatter(style="github-dark")
    rules = formatter.get_style_defs(".codehilite")
    return (
        rules
        + """
.codehilite { background: #0a0f15; }
.codehilite pre { margin: 0; padding: 0; border: 0; background: none; }
pre .codehilite, li .codehilite { background: #0a0f15; }
"""
    )


def render(source: Path) -> Path:
    text = source.read_text(encoding="utf-8")
    md = markdown.Markdown(
        extensions=[
            "fenced_code",
            "codehilite",
            "tables",
            "toc",
            "sane_lists",
            "attr_list",
        ],
        extension_configs={
            "codehilite": {"guess_lang": False, "css_class": "codehilite"},
            # Keep Chinese in anchors instead of the ASCII-only default slugs.
            "toc": {"toc_depth": "2-3", "permalink": False, "slugify": slugify_unicode},
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
        css=CSS + codehilite_css(),
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
