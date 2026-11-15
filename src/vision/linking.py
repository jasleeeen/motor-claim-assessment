"""Assign each detected damage to the car part it sits on, and decide when a human must look at the photographs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

Box = tuple[float, float, float, float]      # x1, y1, x2, y2

SCORE_THRESHOLD = 0.6    
MIN_CONTAINMENT = 0.5 
MARGINAL_CONTAINMENT = 0.8 
COSMETIC = {"Scratch", "Paint chip", "Other damage"}
OVERALL_INSTANCE_RECALL = 0.22
INSTANCE_RECALL_WITH_CLASS = 0.18


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: Box


@dataclass
class VisionResult:
    parts_detected: list[str] = field(default_factory=list)
    damaged_parts: dict[str, list[str]] = field(default_factory=dict)
    unlinked_damage: list[str] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    score_threshold: float = SCORE_THRESHOLD

    def part_names(self) -> list[str]:
        return sorted(self.damaged_parts)


def box_area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def containment(damage: Box, part: Box) -> float:
    """Share of the damage box lying inside the part box."""
    area = box_area(damage)
    if area <= 0:
        return 0.0
    overlap = (max(0.0, min(damage[2], part[2]) - max(damage[0], part[0]))
               * max(0.0, min(damage[3], part[3]) - max(damage[1], part[1])))
    return overlap / area


def link(damages: list[Detection], parts: list[Detection],
         score_threshold: float = SCORE_THRESHOLD,
         min_containment: float = MIN_CONTAINMENT) -> VisionResult:
    """Each damage goes to the part containing most of it.

    Nested panels are the known weakness: a headlight sits inside the bumper's box, so
    a damage inside both is ambiguous. The smaller part wins, because it is the more
    specific claim; ties go to the higher detection score.
    """
    parts = [p for p in parts if p.score >= score_threshold]
    damages = [d for d in damages if d.score >= score_threshold]

    found: dict[str, set[str]] = defaultdict(set)
    unlinked: list[str] = []
    marginal: list[str] = []

    for d in damages:
        best, best_c, best_area = None, 0.0, float("inf")
        for p in parts:
            c = containment(d.box, p.box)
            if c < min_containment:
                continue
            area = box_area(p.box)
            if (c, -area, p.score) > (best_c, -best_area, best.score if best else -1.0):
                best, best_c, best_area = p, c, area
        if best is None:
            unlinked.append(d.label)
        else:
            found[best.label].add(d.label)
            if best_c < MARGINAL_CONTAINMENT:
                marginal.append(f"{d.label} on {best.label} ({best_c:.0%} inside)")

    kinds = {k for v in found.values() for k in v} | set(unlinked)
    reasons: list[str] = []
    if not parts:
        reasons.append("no car parts were found in the photographs, so they may not "
                       "show a vehicle")
    if unlinked:
        reasons.append("damage was found that does not sit on any identified part: "
                       + ", ".join(sorted(set(unlinked))))
    if not damages:
        reasons.append(f"no damage was found, but the detector misses about "
                       f"{1 - OVERALL_INSTANCE_RECALL:.0%} of damage instances, so this "
                       f"is not evidence the vehicle is undamaged")
    if kinds and kinds <= COSMETIC:
        reasons.append("only cosmetic damage was found (scratch or paint chip), which "
                       "this model detects poorly, so the list is probably incomplete")
    if marginal:
        reasons.append("damage straddles a panel edge, so the part may be wrong: "
                       + "; ".join(marginal))

    return VisionResult(
        parts_detected=sorted({p.label for p in parts}),
        damaged_parts={k: sorted(v) for k, v in sorted(found.items())},
        unlinked_damage=sorted(set(unlinked)),
        needs_review=bool(reasons), review_reasons=reasons,
        score_threshold=score_threshold)


def merge(results: list[VisionResult]) -> VisionResult:
    """Combine several photographs of one vehicle. Damage seen in any photograph counts."""
    if not results:
        return VisionResult(needs_review=True,
                            review_reasons=["no photographs were supplied"])
    parts: set[str] = set()
    damaged: dict[str, set[str]] = defaultdict(set)
    unlinked: set[str] = set()
    reasons: list[str] = []
    for n, r in enumerate(results, start=1):
        parts |= set(r.parts_detected)
        for part, kinds in r.damaged_parts.items():
            damaged[part] |= set(kinds)
        unlinked |= set(r.unlinked_damage)
        for reason in r.review_reasons:
            tagged = f"photo {n}: {reason}"
            if tagged not in reasons:
                reasons.append(tagged)
    if damaged:
        reasons = [r for r in reasons if "no damage was found" not in r]
    return VisionResult(sorted(parts), {k: sorted(v) for k, v in sorted(damaged.items())},
                        sorted(unlinked), bool(reasons), reasons)
