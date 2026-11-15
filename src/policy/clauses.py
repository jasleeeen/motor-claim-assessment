"""Split a policy section into individually citable clauses.

The four sample wordings number their clauses four different ways, and three of them
put the marker on a line of its own with the text on the following lines:

    Digit     "1. By fire, explosion..."  
    Bajaj     "a)" / "by fire explosion..." 
    Tata AIG  "(i)" / "by fire, explosion..."
    ICICI     "I." / "By fire explosion..." 

Nesting is inferred from the marker style rather than from a fixed scheme which handles it without hard-coding any insurer's choices.
Each clause gets a stable id such as GENERAL_EXCEPTIONS.3.a. The id is what the model cites and what the validator checks a quoted phrase against.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.policy.extract import Line
from src.policy.sections import Section

_ROMAN = r"(?:x{0,2}(?:ix|iv|v?i{0,3}))"
MARKER_STYLES: list[tuple[str, re.Pattern]] = [
    ("paren_roman",  re.compile(rf"^\(\s*({_ROMAN})\s*\)", re.I)),
    ("paren_letter", re.compile(r"^\(\s*([a-z])\s*\)", re.I)),
    ("paren_number", re.compile(r"^\(\s*(\d{1,2})\s*\)")),
    ("number",       re.compile(r"^(\d{1,2})\s*[.)]")),
    ("roman",        re.compile(r"^([ivxIVX]{1,5})\s*[.)]")),
    ("letter",       re.compile(r"^([a-zA-Z])\s*[.)]")),
]
MAX_MARKER_ONLY_CHARS = 12
MIN_CLAUSE_CHARS = 12 


@dataclass
class Clause:
    id: str
    section: str
    marker: str
    text: str
    page: int
    level: int

    def __str__(self) -> str:
        return f"[{self.id}] {self.text}"


def _match_marker(text: str):
    """Return (style, marker, remainder) when the line opens a clause."""
    for style, pattern in MARKER_STYLES:
        m = pattern.match(text)
        if not m:
            continue
        remainder = text[m.end():].strip(" .:-")
        if remainder or len(text) <= MAX_MARKER_ONLY_CHARS:
            return style, m.group(1), remainder
    return None


def split_clauses(lines: list[Line], section: Section) -> list[Clause]:
    """Clauses of one section, in order, with hierarchical ids."""
    body = lines[section.start:section.end]
    if not body:
        return []

    clauses: list[Clause] = []
    stack: list[str] = [] 
    path: list[str] = []
    pending: Clause | None = None
    used: set[str] = set() 

    def unique(base: str) -> str:
        """Section I of every one of these wordings contains TWO lists numbered from
        1: the insured perils, then the depreciation rates. Without this the second
        list overwrote the first and the cover grant - fire, theft, riot, earthquake,
        flood - vanished from the index entirely."""
        candidate, n = base, 2
        while candidate in used:
            candidate = f"{base}~{n}"
            n += 1
        used.add(candidate)
        return candidate

    def flush():
        nonlocal pending
        if pending is not None:
            pending.text = " ".join(pending.text.split())
            if len(pending.text) >= MIN_CLAUSE_CHARS or not clauses:
                clauses.append(pending)
            elif clauses:
                clauses[-1].text += " " + pending.text
            pending = None

    preamble: list[str] = []
    started = False

    for line in body[1:]:            
        hit = _match_marker(line.text)
        if hit:
            style, marker, remainder = hit
            if not started:
                started = True
                if preamble:
                    clauses.append(Clause(unique(f"{section.kind}.0"), section.kind, "",
                                          " ".join(" ".join(preamble).split()),
                                          body[0].page, 0))
            flush()
            if style in stack:
                depth = stack.index(style)
                del stack[depth + 1:]
                del path[depth:]
            else:
                depth = len(stack)
                stack.append(style)
            path.append(marker)
            clause_id = unique(f"{section.kind}." + ".".join(path))
            pending = Clause(clause_id, section.kind, marker, remainder,
                             line.page, len(path) - 1)
        elif started:
            if pending is not None:
                pending.text += " " + line.text
        else:
            preamble.append(line.text)

    flush()
    if not started and preamble:
        clauses.append(Clause(unique(f"{section.kind}.0"), section.kind, "",
                              " ".join(" ".join(preamble).split()), body[0].page, 0))
    return clauses


def build_index(lines: list[Line], sections: list[Section]) -> list[Clause]:
    out: list[Clause] = []
    for s in sections:
        out.extend(split_clauses(lines, s))
    ids = [c.id for c in out]
    assert len(ids) == len(set(ids)), "clause ids must be unique or clauses are lost"
    return out


def as_lookup(clauses: list[Clause]) -> dict[str, str]:
    """Clause id -> text, which is what the citation validator checks against."""
    return {c.id: c.text for c in clauses}


def render_for_prompt(clauses: list[Clause]) -> str:
    return "\n".join(f"[{c.id}] {c.text}" for c in clauses)
