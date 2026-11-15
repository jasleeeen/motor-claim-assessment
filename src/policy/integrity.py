"""Checks on an uploaded policy document before its clauses are trusted to prevent prompt injections.
   Every Indian private car package policy is built on the same IRDAI standard form, so a Section I that lists none of its perils is not one.
"""
from __future__ import annotations

import re

INSTRUCTION_PATTERNS = [
    r"\bignore\s+(all\s+)?(previous|prior|above|earlier)\b",
    r"\bsystem\s+(instruction|prompt|message)\b",
    r"\b(you|assistant)\s+must\s+(set|return|report|output|approve)\b",
    r"\bdisregard\s+(the\s+)?(above|previous|rules|instructions)\b",
    r"\b(always|never)\s+(approve|reject)\s+(all|every|any)\b",
    r"\bregardless\s+of\s+the\s+(facts|evidence|damage)\b",
    r"^\s*(assistant|system|user)\s*:",
]
# Phrases from the IRDAI standard form that a genuine private car wording carries.
STANDARD_FORM_MARKERS = [
    r"will indemnify the insured against loss or damage",
    r"accidental external means",
    r"shall not be liable",
    r"burglary|housebreaking|theft",
]
MIN_STANDARD_MARKERS = 3


class PolicyIntegrityProblem(ValueError):
    """The uploaded document should not be trusted without a human looking at it."""


def find_injected_instructions(clauses: dict[str, str]) -> list[tuple[str, str]]:
    """Clauses that read like instructions to a machine rather than policy terms."""
    out = []
    for cid, text in clauses.items():
        for pattern in INSTRUCTION_PATTERNS:
            m = re.search(pattern, text, re.I | re.M)
            if m:
                out.append((cid, text[max(0, m.start() - 20):m.end() + 60].strip()))
                break
    return out


def missing_standard_form_markers(clauses: dict[str, str]) -> list[str]:
    blob = " ".join(clauses.values())
    return [p for p in STANDARD_FORM_MARKERS if not re.search(p, blob, re.I)]


def check_policy(clauses: dict[str, str]) -> list[str]:
    """Reasons the document should go to a human. Empty means it looks ordinary."""
    reasons = []
    for cid, snippet in find_injected_instructions(clauses):
        reasons.append(f"clause {cid} contains text addressed to an automated system, "
                       f"not policy terms: “{snippet[:120]}”")
    missing = missing_standard_form_markers(clauses)
    if len(STANDARD_FORM_MARKERS) - len(missing) < MIN_STANDARD_MARKERS:
        reasons.append("the document does not read like a standard private car policy "
                       f"wording (missing: {', '.join(missing)})")
    return reasons
