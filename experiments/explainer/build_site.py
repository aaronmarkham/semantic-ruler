"""Build the static site in docs/ from the two explainer pages.

The pages in this folder are written as fragments (title, styles, body, script)
and fetch their data from a JSON file beside them. This wraps each in a full
HTML document, copies the data, and points the two pages at each other. Serve
docs/ with GitHub Pages (Settings > Pages > main, /docs). Standard library only.

    python -I experiments/explainer/build_site.py .
"""

import re
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
here = root / "experiments" / "explainer"
docs = root / "docs"
docs.mkdir(exist_ok=True)

# source page, its data file, the published name, a one-line description
PAGES = [
    ("explainer.html", "data.json", "index.html",
     "Measuring abstract concepts with a calibrated direction in embedding space, and how often AWS how-to pages send the reader on a detour."),
    ("adventure.html", "trail.json", "detour-map.html",
     "Walk 219 AWS how-to pages as a map: where the prerequisites are, which links deliver, and which are dead ends."),
    ("drift.html", "drift-explainer.json", "drift.html",
     "What a fact store can't hold: measuring semantic drift when documentation is distilled to atomic facts and regenerated."),
]
# The fragments link to each other's claude.ai copies; on the site they link to the neighbouring file.
LINKS = {
    "https://claude.ai/artifact/5GeHhmY5vnF5dBqbkmqaws": "index.html",
    "https://claude.ai/artifact/8W3PY5MkxCgLdcLecq9YTN": "detour-map.html",
}

for src, data, name, description in PAGES:
    page = (here / src).read_text(encoding="utf-8")
    title = re.search(r"<title>(.*?)</title>", page).group(1)
    for url, local in LINKS.items():
        page = page.replace(url, local)
    assert "claude.ai/artifact" not in page, f"{src} still links to an artifact"
    assert f'fetch("{data}")' in page, f"{src} does not load {data}"
    head = (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
            f'<meta name="description" content="{description}">\n'
            f'<meta property="og:title" content="{title}">\n<meta property="og:description" content="{description}">\n'
            f"<style>body{{margin:0}}</style>\n</head>\n<body>\n")
    (docs / name).write_text(head + page.rstrip() + "\n</body>\n</html>\n", encoding="utf-8")
    shutil.copyfile(here / data, docs / data)
    print(f"{name:16s} {(docs / name).stat().st_size / 1024:5.0f} KB   {data:10s} {(docs / data).stat().st_size / 1024:5.0f} KB")

(docs / ".nojekyll").write_text("", encoding="utf-8")   # serve the files as they are
print(f"site: {docs}")
