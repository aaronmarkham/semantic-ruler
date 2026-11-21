"""Turn a cached AWS docs page into an ordered stream of numbered blocks.

Each block is a heading, a prose paragraph or list item (with its links to
other doc pages), or a code block. Stdlib only.
"""

from __future__ import annotations

import html
import json
import re
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin

BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "dt", "dd", "td", "th", "div", "br", "tr"}
SKIP = {"script", "style", "nav", "awsdocs-language-banner", "awsdocs-page-header"}
HEAD = {"h1", "h2", "h3", "h4"}


class StreamParser(HTMLParser):
    def __init__(self, page_url: str):
        super().__init__(convert_charrefs=True)
        self.url = urldefrag(page_url)[0]
        self.blocks: list[dict] = []
        self.in_main = self.skip = self.in_pre = self.heading = 0
        self._buf: list[str] = []
        self._code: list[str] = []
        self._links: list[dict] = []
        self._a: tuple[str, int] | None = None

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._buf = []
        links, self._links = self._links, []
        if not text:
            return
        if self.heading:
            self.blocks.append({"kind": "h", "level": self.heading, "text": text})
        else:
            self.blocks.append({"kind": "p", "text": text, "links": links})

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id") == "main-col-body" or (tag == "main" and not self.in_main):
            self.in_main = 1
        if not self.in_main:
            return
        if tag in SKIP:
            self.skip += 1
        if tag == "pre":
            self._flush()
            self.in_pre += 1
        elif tag in BLOCK and not self.in_pre:
            self._flush()
            if tag in HEAD:
                self.heading = int(tag[1])
        elif tag == "a" and a.get("href") and not self.in_pre:
            self._a = (a["href"], len(self._buf))

    def handle_endtag(self, tag):
        if not self.in_main:
            return
        if tag in SKIP:
            self.skip = max(0, self.skip - 1)
        if tag == "pre" and self.in_pre:
            self.in_pre -= 1
            if not self.in_pre:
                self.blocks.append({"kind": "code", "text": "".join(self._code).strip()})
                self._code = []
        elif tag in BLOCK and not self.in_pre:
            self._flush()
            if tag in HEAD:
                self.heading = 0
        elif tag == "a" and self._a:
            href, start = self._a
            self._a = None
            target = urldefrag(urljoin(self.url, href))[0]
            text = re.sub(r"\s+", " ", "".join(self._buf[start:])).strip()
            if target.startswith("https://docs.aws.amazon.com/") and target != self.url and text:
                self._links.append({"text": text, "url": target})

    def handle_data(self, data):
        if not self.in_main or self.skip:
            return
        (self._code if self.in_pre else self._buf).append(data)


def parse(body: str, page_url: str) -> list[dict]:
    p = StreamParser(page_url)
    p.feed(body)
    p._flush()
    return p.blocks


def policy_summary(code: str) -> str | None:
    """One-line description of a policy JSON block, or None if it is not a policy."""
    if '"Statement"' not in code:
        return None
    text = html.unescape(code)
    try:
        doc = json.JSONDecoder().raw_decode(text, text.find("{"))[0]
        stmts = doc.get("Statement", [])
        stmts = stmts if isinstance(stmts, list) else [stmts]
    except (json.JSONDecodeError, AttributeError, ValueError):
        return "policy JSON (unparsed)"
    principals, actions = set(), set()
    for s in stmts:
        if not isinstance(s, dict):
            continue
        pr = s.get("Principal")
        if pr == "*":
            principals.add("anyone")
        elif isinstance(pr, dict):
            principals.update("service principal" if k == "Service" else "account principal" for k in pr)
        acts = s.get("Action", [])
        for act in acts if isinstance(acts, list) else [acts]:
            if isinstance(act, str):
                actions.add(act.split(":")[0])
    kind = ("resource-based policy naming " + " and ".join(sorted(principals))) if principals else "identity-based policy"
    return f"policy JSON: {kind}; services in actions: {', '.join(sorted(actions)) or 'none'}"


def render(blocks: list[dict], max_chars: int = 14000) -> tuple[str, int]:
    """Numbered text for the LLM. Returns (text, number of blocks included)."""
    out, used = [], 0
    for i, b in enumerate(blocks):
        if b["kind"] == "h":
            line = f"[{i}] {'#' * b['level']} {b['text']}"
        elif b["kind"] == "code":
            line = f"[{i}] <{policy_summary(b['text']) or 'code or command example'}>"
        else:
            links = "".join(f" (link: {l['text']} -> {l['url']})" for l in b["links"][:4])
            line = f"[{i}] {b['text'][:700]}{links}"
        if used + len(line) > max_chars:
            return "\n".join(out), i
        out.append(line)
        used += len(line) + 1
    return "\n".join(out), len(blocks)
