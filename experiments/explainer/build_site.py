"""Build the static site in docs/ from the two explainer pages.

The pages in this folder are written as fragments (title, styles, body, script)
and fetch their data from a JSON file beside them. This wraps each in a full
HTML document, copies the data, points the pages at each other, and puts one
shared navigation bar at the top of each: the other pages, a menu of the
current page's numbered sections, and the repository. Serve docs/ with GitHub
Pages (Settings > Pages > main, /docs). Standard library only.

    python -I experiments/explainer/build_site.py .
"""

import html
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
# Short names for the navigation bar, by published name.
SHORT = {"index.html": "Rulers and Detours", "detour-map.html": "The Detour Map", "drift.html": "Drift"}
REPO = "https://github.com/aaronmarkham/semantic-ruler"
# The fragments link to each other's claude.ai copies; on the site they link to the neighbouring file.
LINKS = {
    "https://claude.ai/artifact/5GeHhmY5vnF5dBqbkmqaws": "index.html",
    "https://claude.ai/artifact/8W3PY5MkxCgLdcLecq9YTN": "detour-map.html",
}

# Each numbered section opens with an eyebrow ("01 · The paper"). Give the ones that have no id one,
# so the navigation bar can link to them, and collect (id, label) for the menu.
SECTION = re.compile(r'<section(?: id="([^"]+)")?>\n<p class="eyebrow">(.*?)</p>')


def sections(page):
    found = []

    def mark(m):
        sid, label = m.group(1), re.sub(r"<[^>]+>", "", m.group(2))
        if not sid:
            sid = "s" + re.match(r"\d+", label).group(0)
        found.append((sid, html.unescape(label)))
        return f'<section id="{sid}">\n<p class="eyebrow">{m.group(2)}</p>'

    return SECTION.sub(mark, page), found


NAV_STYLE = """
/* Shared navigation bar. Colours and faces come from each page's own palette, with fallbacks. */
:root{--sn-h:46px}
html{scroll-padding-top:calc(var(--sn-h) + 14px)}
.sn{position:sticky;top:0;z-index:30;display:flex;align-items:center;gap:4px 18px;min-height:var(--sn-h);
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));
  background:var(--paper,#EFF1EE);border-bottom:1px solid var(--rule,#B7BFBA);
  font:12px/1.4 var(--mono,ui-monospace,Consolas,monospace);letter-spacing:.06em;color:var(--muted,#55635F)}
.sn a{color:var(--ink,#131C21);text-decoration:none}
.sn a:hover,.sn summary:hover{color:var(--invariant,var(--ok,#3B3190))}
.sn-brand{font-weight:500;letter-spacing:.1em;text-transform:uppercase;margin-right:auto;white-space:nowrap}
.sn-pages{display:flex;gap:16px;margin:0;padding:0;list-style:none}
.sn-pages a{white-space:nowrap;padding:14px 0;display:inline-block;border-bottom:2px solid transparent}
.sn-pages a[aria-current]{color:var(--invariant,var(--ok,#3B3190));border-bottom-color:currentColor}
.sn-gh{white-space:nowrap}
.sn-menu{position:relative}
.sn-menu summary{list-style:none;cursor:pointer;white-space:nowrap;padding:14px 0;color:var(--ink,#131C21);display:inline-flex;align-items:center;gap:6px}
.sn-menu summary::-webkit-details-marker{display:none}
.sn-menu summary::after{content:"";width:6px;height:6px;border:solid currentColor;border-width:0 1.5px 1.5px 0;transform:translateY(-2px) rotate(45deg)}
.sn-menu[open] summary::after{transform:translateY(1px) rotate(-135deg)}
.sn-drop{position:absolute;right:0;top:calc(100% - 1px);min-width:260px;max-width:min(360px,calc(100vw - 32px));max-height:calc(100vh - var(--sn-h) - 16px);overflow:auto;
  background:var(--paper,#EFF1EE);border:1px solid var(--rule,#B7BFBA);border-radius:0 0 6px 6px;box-shadow:0 10px 24px -12px rgba(19,28,33,.45);padding:8px 0 10px}
.sn-h{margin:10px 16px 4px;font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted,#55635F)}
.sn-drop ul{margin:0;padding:0;list-style:none}
.sn-drop a{display:block;padding:8px 16px;letter-spacing:.03em;font-size:13px}
.sn-drop a:hover{background:var(--sunk,#E7EAE7)}
.sn-drop .sn-pages{display:block}
.sn-drop .sn-pages a{padding:8px 16px;border:0}
.sn-drop a[aria-current]{color:var(--invariant,var(--ok,#3B3190))}
.sn-drop .sn-gh{display:block;padding:8px 16px;margin-top:4px;border-top:1px solid var(--rule,#B7BFBA);padding-top:12px}
.sn-only-narrow{display:none}
@media (max-width:719px){
  .sn > .sn-pages,.sn > .sn-gh,.sn-only-wide{display:none}
  .sn-only-narrow{display:block}
}
@media (min-width:720px){.sn-drop > .sn-only-narrow{display:none}}
/* Elements the pages pin to the top of the window sit below the bar. */
body #deep .story{top:calc(var(--sn-h) + 8px)}
@media (min-width:680px){body .walk .story{top:calc(var(--sn-h) + 12px)}}
@media print{.sn{display:none}}
"""

NAV_SCRIPT = """<script>
(() => {
  const m = document.querySelector(".sn-menu"); if (!m) return;
  const close = () => m.removeAttribute("open");
  m.addEventListener("click", e => { if (e.target.closest("a")) close(); });
  document.addEventListener("click", e => { if (m.open && !m.contains(e.target)) close(); });
  document.addEventListener("keydown", e => { if (e.key === "Escape" && m.open) { close(); m.querySelector("summary").focus(); } });
})();
</script>"""


def nav(name, found):
    def pages(cls=""):
        items = []
        for _, _, n, _ in PAGES:
            current = ' aria-current="page"' if n == name else ""
            items.append(f'<li><a href="{n}"{current}>{html.escape(SHORT[n])}</a></li>')
        return f'<ul class="sn-pages{cls}">{"".join(items)}</ul>'
    secs = "".join(f'<li><a href="#{sid}">{html.escape(label)}</a></li>' for sid, label in found)
    gh = f'<a class="sn-gh" href="{REPO}">GitHub</a>'
    return (f'<nav class="sn" aria-label="Site">\n<a class="sn-brand" href="index.html">semantic-ruler</a>\n{pages()}\n'
            f'<details class="sn-menu">\n<summary><span class="sn-only-wide">On this page</span><span class="sn-only-narrow">Menu</span></summary>\n'
            f'<div class="sn-drop">\n<p class="sn-h sn-only-narrow">Pages</p>\n{pages(" sn-only-narrow")}\n'
            f'<p class="sn-h">On this page</p>\n<ul>{secs}</ul>\n{gh}\n</div>\n</details>\n{gh}\n</nav>\n{NAV_SCRIPT}\n')


for src, data, name, description in PAGES:
    page = (here / src).read_text(encoding="utf-8")
    title = re.search(r"<title>(.*?)</title>", page).group(1)
    for url, local in LINKS.items():
        page = page.replace(url, local)
    assert "claude.ai/artifact" not in page, f"{src} still links to an artifact"
    assert f'fetch("{data}")' in page, f"{src} does not load {data}"
    page, found = sections(page)
    assert found, f"{src} has no numbered sections"
    head = (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
            f'<meta name="description" content="{description}">\n'
            f'<meta property="og:title" content="{title}">\n<meta property="og:description" content="{description}">\n'
            f"<style>body{{margin:0}}{NAV_STYLE}</style>\n</head>\n<body>\n" + nav(name, found))
    (docs / name).write_text(head + page.rstrip() + "\n</body>\n</html>\n", encoding="utf-8")
    shutil.copyfile(here / data, docs / data)
    print(f"{name:16s} {(docs / name).stat().st_size / 1024:5.0f} KB   {data:10s} {(docs / data).stat().st_size / 1024:5.0f} KB")

(docs / ".nojekyll").write_text("", encoding="utf-8")   # serve the files as they are
print(f"site: {docs}")
