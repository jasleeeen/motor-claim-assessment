"""A deterministic reader with the same interface as the Groq one, which lets the whole 
pipeline be tested offline, with no API key and no rate limit.
It is deliberately simple: keyword rules over the same clause index the model sees. It
has no idea what a clause means, so it should lose on wording it has not been taught.
"""
from __future__ import annotations
import re
from src.claim.decision import Finding, Scope, validate_citation

PERILS = {
    "fire": r"\bfire\b|explosion|self[- ]?ignition|lightning",
    "theft": r"burglary|housebreaking|theft",
    "riot": r"riot|strike",
    "earthquake": r"earthquake",
    "flood": r"flood|typhoon|hurricane|storm|tempest|inundation|cyclone|hailstorm|frost",
    "accident": r"accidental external means",
    "malicious": r"malicious",
    "terrorism": r"terrorist",
    "transit": r"in transit by road",
    "landslide": r"landslide|rockslide",
}
EXCLUSION_TRIGGERS = {
    "intoxicated": (r"intoxicat|under the influence", Scope.WHOLE_CLAIM),
    "no_valid_licence": (r"driving licen|duly licensed", Scope.WHOLE_CLAIM),
    "outside_geographical_area": (r"geographical area", Scope.WHOLE_CLAIM),
    "commercial_use": (r"limitations as to use", Scope.WHOLE_CLAIM),
    "war_or_nuclear": (r"nuclear|war|invasion|mutiny", Scope.WHOLE_CLAIM),
    "wear_and_tear": (r"wear and tear", Scope.SOME_PARTS),
    "mechanical_breakdown": (r"mechanical or electrical breakdown", Scope.SOME_PARTS),
}
PART_MATERIALS = {
    r"bumper|fender|grille|trim": r"rubber/?nylon/?plastic|plastic parts",
    r"window|windshield|windscreen|glass": r"parts made of glass|glass",
    r"tyre|tube": r"tyres? and tubes?",
}


def _first(clauses: dict[str, str], pattern: str, section: str | None = None):
    for cid, text in clauses.items():
        if section and not cid.startswith(section):
            continue
        if re.search(pattern, text, re.I):
            return cid, text
    return None, None


def _quote(text: str, pattern: str, words: int = 12) -> str:
    """A short verbatim span around the match, so the citation validator can check it."""
    m = re.search(pattern, text, re.I)
    if not m:
        return " ".join(text.split()[:words])
    tokens = text.split()
    upto = len(text[:m.start()].split())
    start = max(0, upto - 2)
    return " ".join(tokens[start:start + words])


def read_policy(incident: dict, damaged_parts: list[str], clauses: dict[str, str],
                clause_text: str = "", **_):
    """Same return shape as src/claim/llm.read_policy."""
    cause = (incident.get("cause") or "").lower()

    pattern = PERILS.get(cause)
    if pattern:
        cid, text = _first(clauses, pattern, section="SECTION_I")
        cause_finding = Finding(clause_id=cid, quote=_quote(text or "", pattern),
                                detail="insured" if cid else "not_insured")
    else:
        cid, text = _first(clauses, r"will indemnify the insured", section="SECTION_I")
        cause_finding = Finding(clause_id=cid, quote=_quote(text or "", r"indemnify"),
                                detail="not_insured")
    cause_finding = validate_citation(cause_finding, clauses)

    exclusions = []
    for flag, (pattern, scope) in EXCLUSION_TRIGGERS.items():
        if not incident.get(flag):
            continue
        cid, text = _first(clauses, pattern)
        if cid:
            exclusions.append(validate_citation(
                Finding(clause_id=cid, quote=_quote(text, pattern), scope=scope,
                        parts=list(damaged_parts) if scope is Scope.SOME_PARTS else []),
                clauses))

    TYRE = r"tyre|tube|wheel"
    if damaged_parts and all(re.search(TYRE, p, re.I) for p in damaged_parts):
        cid, text = _first(clauses, r"Damage to tyres and tubes unless", section="SECTION_I")
        if cid:
            exclusions.append(validate_citation(
                Finding(clause_id=cid, quote=_quote(text, r"Damage to tyres and tubes"),
                        scope=Scope.WHOLE_CLAIM), clauses))

    limits = []
    for part_pattern, clause_pattern in PART_MATERIALS.items():
        hit = [p for p in damaged_parts if re.search(part_pattern, p, re.I)]
        if not hit:
            continue
        cid, text = _first(clauses, clause_pattern, section="SECTION_I")
        if cid:
            limits.append(validate_citation(
                Finding(clause_id=cid, quote=_quote(text, clause_pattern),
                        scope=Scope.SOME_PARTS, parts=hit), clauses))

    return cause_finding, exclusions, limits, "baseline: keyword rules, no model"
