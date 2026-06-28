"""Extract the text of rulebook.pdf into docs/rulebook.md.

The PDF has a real text layer (not a scan), so we just pull text in reading
order and drop images/vector art entirely. A light heading heuristic promotes
lines whose font is noticeably larger than the page's body text into markdown
headings; everything else is emitted as plain paragraphs. Pages are separated
by `<!-- page N -->` markers so the output stays navigable.

Repeating page boilerplate is filtered out: the running header (any line whose
text recurs on a large fraction of pages) and the bare page-number footer (a
line whose only content is the current page number).

Usage:
    python extract_rulebook.py [input.pdf] [output.md]
"""

import sys
from collections import Counter

import fitz  # PyMuPDF


def round_size(size: float) -> float:
    return round(size, 1)


def body_font_size(doc: "fitz.Document") -> float:
    """Most common span font size across the document = body text size."""
    counts: Counter = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:  # 0 = text block; skip images (type 1)
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        counts[round_size(span["size"])] += len(text)
    if not counts:
        return 0.0
    return counts.most_common(1)[0][0]


def heading_level(line_size: float, body: float) -> int:
    """Return markdown heading level (2/3) or 0 for body text."""
    if body <= 0:
        return 0
    ratio = line_size / body
    if ratio >= 1.5:
        return 2
    if ratio >= 1.2:
        return 3
    return 0


def line_size_and_text(line: dict) -> tuple[float, str]:
    """Max span size on the line + its concatenated text."""
    parts = [span["text"] for span in line["spans"]]
    sizes = [span["size"] for span in line["spans"] if span["text"].strip()]
    text = "".join(parts).strip()
    size = round_size(max(sizes)) if sizes else 0.0
    return size, text


def repeating_boilerplate(doc: "fitz.Document", min_pages: int) -> set[str]:
    """Line texts that recur on >= min_pages distinct pages (running headers)."""
    pages_with: dict[str, set[int]] = {}
    for i, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                _, text = line_size_and_text(line)
                if text:
                    pages_with.setdefault(text, set()).add(i)
    return {text for text, pages in pages_with.items() if len(pages) >= min_pages}


def is_boilerplate(text: str, page_no: int, headers: set[str]) -> bool:
    """True for the running header or the bare page-number footer."""
    return text in headers or text == str(page_no)


def extract_page(
    page: "fitz.Page", page_no: int, body: float, headers: set[str]
) -> list[str]:
    out: list[str] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # skip image blocks
            continue
        for line in block["lines"]:
            size, text = line_size_and_text(line)
            if not text or is_boilerplate(text, page_no, headers):
                continue
            level = heading_level(size, body)
            if level:
                out.append("#" * level + " " + text)
            else:
                out.append(text)
        out.append("")  # blank line between blocks
    return out


def collapse_blanks(lines: list[str]) -> list[str]:
    result: list[str] = []
    blank = False
    for ln in lines:
        if ln.strip() == "":
            if not blank:
                result.append("")
            blank = True
        else:
            result.append(ln)
            blank = False
    return result


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "rulebook.pdf"
    dst = sys.argv[2] if len(sys.argv) > 2 else "docs/rulebook.md"

    doc = fitz.open(src)
    body = body_font_size(doc)
    # A running header recurs on most pages; require it on >=1/4 of them so we
    # never mistake genuinely repeated content (e.g. diagram labels) for chrome.
    headers = repeating_boilerplate(doc, min_pages=max(5, doc.page_count // 4))
    print(f"{src}: {doc.page_count} pages, body font size ~{body}")
    print(f"filtering {len(headers)} running-header line(s) + page numbers")

    lines: list[str] = [f"<!-- extracted from {src} ({doc.page_count} pages) -->", ""]
    for i, page in enumerate(doc, start=1):
        lines.append(f"<!-- page {i} -->")
        lines.append("")
        lines.extend(extract_page(page, i, body, headers))

    text = "\n".join(collapse_blanks(lines)).rstrip() + "\n"
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(text)

    words = len(text.split())
    print(f"wrote {dst}: {len(text.splitlines())} lines, ~{words} words")


if __name__ == "__main__":
    main()
