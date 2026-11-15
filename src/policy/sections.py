"""Locate the structural sections of a motor policy wording.

Only a few sections can decide an own-damage claim: Section I (what is covered and
its own exclusions), the General Exceptions, and the Deductible. Sections II and III
cover third-party liability and personal accident, and for ICICI the endorsements,
ombudsman list and add-ons make up most of the document.

That matters because the free Groq tier allows 8,000 tokens a minute and these
wordings run from much more, so the whole document cannot be sent. Selecting
by section rather than by similarity search means nothing that could decide the claim
is dropped: the exclusions always travel with the coverage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.policy.extract import Line

_ROMAN = r"(?:I{1,3}|1|2|3)"
ANCHORS: list[tuple[str, re.Pattern]] = [
    ("SECTION_I",   re.compile(rf"^SECTION\s*[-–—]?\s*(?:I|1)\s*[.:–—-]?\s*"
                               rf"(?:LOSS\b.*)?$", re.I)),
    ("SECTION_II",  re.compile(rf"^SECTION\s*[-–—]?\s*(?:II|2)\s*[.:–—-]?\s*"
                               rf"(?:LIABILITY\b.*)?$", re.I)),
    ("SECTION_III", re.compile(rf"^SECTION\s*[-–—]?\s*(?:III|3)\s*[.:–—-]?\s*"
                               rf"(?:PERSONAL\b.*)?$", re.I)),
    ("GENERAL_EXCEPTIONS", re.compile(r"^GENERAL\s+EXCEPTIONS?\b.*$", re.I)),
    ("DEDUCTIBLE",  re.compile(r"^DEDUCTIBLE\s*[:.]?\s*(?:\(Applicable.*)?$", re.I)),
    ("CONDITIONS",  re.compile(r"^CONDITIONS?\s*[:.]?\s*$", re.I)),
    ("NO_CLAIM_BONUS", re.compile(r"^NO\s+CLAIM\s+BONUS\b.*$", re.I)),
    ("CANCELLATION", re.compile(r"^CANCELLATION\s*[:.]?\s*$", re.I)),
    # For cross-references, the clauses they point at have to travel with them.
    ("DRIVERS_CLAUSE", re.compile(r"^(CHAPTER\s+\d+\s*[-–—.:]\s*)?DRIVER.?S\s+CLAUSE\b.*$", re.I)),
    ("LIMITATIONS_AS_TO_USE",
     re.compile(r"^(CHAPTER\s+\d+\s*[-–—.:]\s*)?LIMITATIONS?\s+AS\s+TO\s+USE\b.*$", re.I)),
    ("CHAPTER",     re.compile(r"^CHAPTER\s+\d+\s*[-–—.:].*$", re.I)),
    ("GRIEVANCE",   re.compile(r"^(CUSTOMER\s+)?GRIEVANCE.*$", re.I)),
    ("OMBUDSMAN",   re.compile(r"^.*INSURANCE\s+OMBUDSMAN.*$", re.I)),
    ("ANNEXURE",    re.compile(r"^ANNEXURE\b.*$", re.I)),
]

# Sections sent to the model for a vehicle-damage claim
RELEVANT = ("SECTION_I", "GENERAL_EXCEPTIONS", "LIMITATIONS_AS_TO_USE",
            "DRIVERS_CLAUSE", "DEDUCTIBLE", "NO_CLAIM_BONUS")

MAX_HEADING_CHARS = 90
TOC_MIN_ANCHORS = 4    
TOC_MAX_GAP_LINES = 8 


@dataclass
class Section:
    kind: str
    heading: str
    start: int      
    end: int      
    page: int

    @property
    def n_lines(self) -> int:
        return self.end - self.start


def _is_heading(text: str) -> bool:
    return len(text) <= MAX_HEADING_CHARS


def _drop_contents_listings(hits: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    """Remove runs of anchors that are a table of contents rather than the sections. """
    if len(hits) < TOC_MIN_ANCHORS:
        return hits
    drop: set[int] = set()
    start = 0
    while start < len(hits):
        end = start
        while (end + 1 < len(hits)
               and hits[end + 1][0] - hits[end][0] <= TOC_MAX_GAP_LINES):
            end += 1
        if end - start + 1 >= TOC_MIN_ANCHORS:
            run_kinds = {hits[i][1] for i in range(start, end + 1)}
            later_kinds = {k for i, k, _ in hits[end + 1:]}
            if run_kinds <= later_kinds:                 # every kind reappears later
                drop.update(range(start, end + 1))
            start = end + 1
        else:
            start += 1
    return [h for i, h in enumerate(hits) if i not in drop]


def find_sections(lines: list[Line]) -> list[Section]:
    """Every anchor found, in document order, each running to the next anchor."""
    hits: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        if not _is_heading(line.text):
            continue
        for kind, pattern in ANCHORS:
            if pattern.match(line.text):
                hits.append((i, kind, line.text))
                break

    hits = _drop_contents_listings(hits)

    # Merge consecutive anchors of the same kind: ICICI has "NO CLAIM BONUS :" on one
    # line and the real text under a second anchor immediately after, which would
    # otherwise give a one-line section.
    merged: list[tuple[int, str, str]] = []
    for hit in hits:
        if merged and merged[-1][1] == hit[1]:
            continue
        merged.append(hit)

    sections = []
    for n, (i, kind, text) in enumerate(merged):
        end = merged[n + 1][0] if n + 1 < len(merged) else len(lines)
        sections.append(Section(kind, text, i, end, lines[i].page))
    return sections


def select_relevant(lines: list[Line], sections: list[Section] | None = None,
                    kinds: tuple[str, ...] = RELEVANT) -> list[Section]:
    """The sections that can decide a vehicle-damage claim.
    Where a kind appears more than once, the first occurrence after the contents is taken,
    which is the one in the main standard form.
    """
    sections = sections if sections is not None else find_sections(lines)
    chosen: dict[str, Section] = {}
    for s in sections:
        if s.kind in kinds and s.kind not in chosen:
            chosen[s.kind] = s
    return [chosen[k] for k in kinds if k in chosen]


def section_text(lines: list[Line], section: Section) -> str:
    return "\n".join(l.text for l in lines[section.start:section.end])
