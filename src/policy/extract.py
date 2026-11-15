"""Turn a policy PDF into clean, ordered lines of text.

Problems in the samples this handles:

1. Two-column layouts (ICICI, Tata AIG). PyMuPDF's sort=True interleaves the
   columns, so blocks are grouped into columns by x-position and each
   column is read top to bottom.
2. Running headers and footers land mid-sentence ("... Toll Free No.) Page 2 of 13
   UIN: IRDAN113... The Company shall not be liable ..."), splitting clauses. They are
   detected by repetition across pages rather than by hard-coded insurer patterns.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ALWAYS_DROP = re.compile(
    r"^(page\s+\d+\s*(of\s*\d+)?|\d+\s*\|\s*page|back to index\s*\|?\s*\d*|index\s*\|?\s*\d*"
    r"|uin\s*[:\-].*|irdai?\s+reg(istration)?\.?\s*no.*|cin\s*[:\-].*|toll\s*free.*"
    r"|www\.\S+|\S+@\S+\.\S+)\s*$",
    re.IGNORECASE,
)
# A line that is nothing but a number is a page number only at the edge of a page.
# Applied everywhere it deleted real values: a year of manufacture, depreciation percentage, a table cell.
BARE_NUMBER = re.compile(r"^\d{1,4}$")
EDGE_FOR_BARE_NUMBER = 2
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st",
             "“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", " ": " "}
MIN_COLUMN_BLOCKS = 3        # blocks needed on each side before a page counts as two-column
HEADER_PAGE_FRACTION = 0.5   # a line repeated on this share of pages is header/footer
EDGE_LINES = 8               # lines at each page edge examined for header/footer (headers run to 6)


@dataclass
class Line:
    page: int               
    text: str


def clean_text(s: str) -> str:
    """Normalise ligatures and unicode punctuation so quotes can be matched exactly."""
    for bad, good in LIGATURES.items():
        s = s.replace(bad, good)
    s = unicodedata.normalize("NFKC", s)
    return " ".join(s.split())


def _page_blocks_in_reading_order(page):
    """Text blocks in true reading order, handling one and two-column pages."""
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    if not blocks:
        return []
    mid = page.rect.width / 2
    left = [b for b in blocks if b[2] <= mid + 5]     
    right = [b for b in blocks if b[0] >= mid - 5]    
    if len(left) >= MIN_COLUMN_BLOCKS and len(right) >= MIN_COLUMN_BLOCKS:
        spanning = [b for b in blocks if b not in left and b not in right]
        return (sorted(spanning, key=lambda b: b[1])
                + sorted(left, key=lambda b: b[1])
                + sorted(right, key=lambda b: b[1]))
    return sorted(blocks, key=lambda b: (round(b[1], 1), b[0]))


def _find_repeated_furniture(pages: list[list[str]]) -> set[str]:
    """Detected by repetition, so no insurer-specific patterns are needed."""
    if len(pages) < 3:
        return set()
    counts: Counter = Counter()
    for lines in pages:
        edge = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        counts.update({re.sub(r"\d+", "#", l) for l in edge})
    threshold = max(2, int(len(pages) * HEADER_PAGE_FRACTION))
    return {k for k, v in counts.items() if v >= threshold}


def extract_lines(pdf_path: str | Path) -> list[Line]:
    """Clean, ordered lines with their page numbers."""
    import pymupdf
    with pymupdf.open(pdf_path) as doc:
        raw_pages = []
        for page in doc:
            lines = []
            for block in _page_blocks_in_reading_order(page):
                for piece in block[4].splitlines():
                    cleaned = clean_text(piece)
                    if cleaned:
                        lines.append(cleaned)
            raw_pages.append(lines)

    furniture = _find_repeated_furniture(raw_pages)
    out: list[Line] = []
    for page_no, lines in enumerate(raw_pages, start=1):
        last = len(lines) - 1
        for i, line in enumerate(lines):
            if ALWAYS_DROP.match(line):
                continue
            at_very_edge = i < EDGE_FOR_BARE_NUMBER or i > last - EDGE_FOR_BARE_NUMBER
            if BARE_NUMBER.match(line):
                # Digits are masked before header/footer matching, so a bare number would
                # otherwise collapse onto "page 7" and vanish - taking a year of
                # manufacture or a table figure with it. Position decides instead.
                if at_very_edge:
                    continue
            elif re.sub(r"\d+", "#", line) in furniture:
                continue
            out.append(Line(page_no, line))
    return out


def extraction_report(pdf_path: str | Path) -> dict:
    """Numbers to check an extraction by, rather than trusting it."""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        pages = len(doc)
        raw_chars = sum(len(p.get_text("text")) for p in doc)
    lines = extract_lines(pdf_path)
    kept = sum(len(l.text) for l in lines)
    return {"file": Path(pdf_path).name, "pages": pages, "raw_chars": raw_chars,
            "kept_chars": kept, "dropped_pct": round(100 * (1 - kept / max(raw_chars, 1)), 1),
            "lines": len(lines), "approx_tokens": kept // 4}
