"""Web page to plain text with the standard library (M4.1).

Keeps what a reader would read: headings (as `#` lines), paragraphs, list items, table
rows. Drops scripts, styles, menus, headers, footers, forms and the like. When the page
marks its main content (`<article>` or `<main>`), only that is kept.
"""

import re
from html.parser import HTMLParser

SKIPPED = frozenset(
    {
        "script",
        "style",
        "noscript",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "svg",
        "template",
        "iframe",
        "button",
        "select",
        "canvas",
        "dialog",
    }
)
NEVER_TEXT = frozenset({"script", "style", "noscript", "svg", "template"})
"""Skipped even when the page's structure is broken."""
BLOCKS = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "section",
        "article",
        "main",
        "table",
        "ul",
        "ol",
        "blockquote",
        "pre",
        "dd",
        "dt",
        "figcaption",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }
)
VOID = frozenset({"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "wbr"})
MAIN = frozenset({"article", "main"})
MIN_MAIN_CHARS = 200
"""A main section shorter than this is probably not the content (a teaser, a card)."""


class _Text(HTMLParser):
    def __init__(self, skipped: frozenset[str] = SKIPPED) -> None:
        super().__init__(convert_charrefs=True)
        self.skipped = skipped
        self.title = ""
        self.all: list[str] = []
        self.main: list[str] = []
        self._skip = 0
        self._main = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in VOID:
            if tag in ("br", "hr"):
                self._add("\n")
            return
        if tag in self.skipped:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        if tag in MAIN:
            self._main += 1
        if tag in BLOCKS:
            self._add("\n")
            if tag[0] == "h" and tag[1:].isdigit():
                self._add("#" * int(tag[1]) + " ")
            elif tag == "li":
                self._add("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.skipped and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        if tag in MAIN and self._main:
            self._main -= 1
        if tag in BLOCKS:
            self._add("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        else:
            self._add(data)

    def _add(self, text: str) -> None:
        if self._skip:
            return
        self.all.append(text)
        if self._main:
            self.main.append(text)


def page_text(html: str) -> tuple[str, str]:
    """(title, readable text) of an HTML page."""
    title, body = _parse(html, SKIPPED)
    if not body:  # an unclosed <header> or <button> hid the whole page: skip less
        title, body = _parse(html, NEVER_TEXT)
    return title, body


def _parse(html: str, skipped: frozenset[str]) -> tuple[str, str]:
    parser = _Text(skipped)
    parser.feed(html)
    parser.close()
    main = _tidy("".join(parser.main))
    body = main if len(main) >= MIN_MAIN_CHARS else _tidy("".join(parser.all))
    return " ".join(parser.title.split()), body


def _tidy(text: str) -> str:
    """One line per block, no blank lines (every token counts in a small window)."""
    lines = (re.sub(r"[ \t\u00a0]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line.strip("-# "))
