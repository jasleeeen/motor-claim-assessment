"""Claim-level checks, citation validation and the decision rules.

The model never decides the outcome. It reads clauses and reports findings; the rules
here turn findings into APPROVED / PARTIALLY_APPROVED / REJECTED / NEEDS_MANUAL_REVIEW,
so every outcome is reproducible and explainable without re-running the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

MAX_QUOTE_WORDS = 25

# Add-ons live on the SCHEDULE; their wording is not in the policy document, so the
# depreciation clause still reads as if it applies. Without this the system quietly
# under-pays every customer who bought depreciation cover.
DEPRECIATION_WAIVERS = {"parts depreciation protect", "zero depreciation",
                        "nil depreciation", "depreciation shield"}
CTL_FRACTION = 0.75


class Outcome(str, Enum):
    APPROVED = "APPROVED"
    PARTIALLY_APPROVED = "PARTIALLY_APPROVED"
    REJECTED = "REJECTED"
    NEEDS_MANUAL_REVIEW = "NEEDS_MANUAL_REVIEW"


class Scope(str, Enum):
    WHOLE_CLAIM = "whole_claim"
    SOME_PARTS = "some_parts"


@dataclass
class Finding:
    """One thing the model says it read in the policy, with where it read it."""
    clause_id: str | None
    quote: str | None = None
    detail: str = ""
    scope: Scope = Scope.WHOLE_CLAIM
    parts: list[str] = field(default_factory=list)
    citation_valid: bool = False
    citation_problem: str | None = None


@dataclass
class Decision:
    outcome: Outcome
    reasons: list[str] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


def normalise(text: str) -> str:
    """Whitespace-insensitive comparison.

    Essential, not cosmetic: clauses wrap mid-phrase in the two-column wordings, so a
    quote spanning a line break would otherwise fail validation and send every real
    claim to manual review.
    """
    return re.sub(r"\s+", " ", text or "").strip().lower()


def validate_citation(finding: Finding, clauses: dict[str, str]) -> Finding:
    """A citation counts only if the clause exists and the quote is really in it."""
    if finding.clause_id is None:
        finding.citation_valid = False
        finding.citation_problem = "no clause cited"
        return finding
    if finding.clause_id not in clauses:
        finding.citation_valid = False
        finding.citation_problem = f"clause {finding.clause_id!r} is not in this policy"
        return finding
    if finding.quote:
        if len(finding.quote.split()) > MAX_QUOTE_WORDS:
            finding.citation_valid = False
            finding.citation_problem = f"quote longer than {MAX_QUOTE_WORDS} words"
            return finding
        if normalise(finding.quote) not in normalise(clauses[finding.clause_id]):
            finding.citation_valid = False
            finding.citation_problem = "quoted words do not appear in the cited clause"
            return finding
    finding.citation_valid = True
    finding.citation_problem = None
    return finding


@dataclass
class ClaimLevelResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def check_claim_level(schedule: dict, incident_date: date) -> ClaimLevelResult:
    """Facts that settle a claim before any clause is read."""
    reasons: list[str] = []
    start, end = schedule.get("period_from"), schedule.get("period_to")
    if not start or not end:
        return ClaimLevelResult(False, ["the schedule has no period of insurance"])

    if incident_date < start:
        reasons.append(f"the incident on {incident_date:%d %b %Y} is before cover began "
                       f"on {start:%d %b %Y}")
    elif incident_date > end:
        reasons.append(f"the incident on {incident_date:%d %b %Y} is after cover ended "
                       f"on {end:%d %b %Y}")

    cover = schedule.get("cover_type")
    if cover == "LIABILITY_ONLY":
        reasons.append("the policy is Liability Only, which does not cover damage to "
                       "the insured's own vehicle")
    elif cover is None:
        reasons.append("the schedule does not state the type of cover")

    return ClaimLevelResult(not reasons, reasons)


def decide(schedule: dict, incident_date: date, damaged_parts: list[str],
           cause_insured: Finding | None, exclusions: list[Finding],
           limits: list[Finding], vision_needs_review: bool = False,
           vision_review_reasons: list[str] | None = None) -> Decision:
    """Turn validated findings into an outcome.

    Order matters. A policy that was not in force cannot be approved however the
    clauses read, so the schedule is checked first.
    """
    reasons: list[str] = []
    cited: list[str] = []

    gate = check_claim_level(schedule, incident_date)
    if not gate.passed:
        return Decision(Outcome.REJECTED, gate.reasons,
                        detail={"stage": "claim_level"})

    bad = [f for f in [cause_insured, *exclusions, *limits]
           if f is not None and not f.citation_valid]
    if bad:
        return Decision(
            Outcome.NEEDS_MANUAL_REVIEW,
            [f"a cited clause could not be verified: {f.citation_problem}" for f in bad],
            detail={"stage": "citation_check"})

    if not damaged_parts:
        return Decision(Outcome.NEEDS_MANUAL_REVIEW,
                        ["no damaged parts were identified from the photographs"],
                        detail={"stage": "damage"})

    if vision_needs_review:
        return Decision(Outcome.NEEDS_MANUAL_REVIEW,
                        list(vision_review_reasons or ["the photographs need a human check"]),
                        detail={"stage": "vision"})

    if cause_insured is None:
        return Decision(Outcome.NEEDS_MANUAL_REVIEW,
                        ["the policy was not read for whether the cause is insured"],
                        detail={"stage": "cause"})
    if cause_insured.detail == "not_insured":
        reasons.append(f"the cause of damage is not an insured peril: "
                       f"{cause_insured.quote or cause_insured.clause_id}")
        cited.append(cause_insured.clause_id)
        return Decision(Outcome.REJECTED, reasons, cited, {"stage": "cause"})
    cited.append(cause_insured.clause_id)

    whole = [f for f in exclusions if f.scope == Scope.WHOLE_CLAIM]
    if whole:
        for f in whole:
            reasons.append(f"excluded by {f.clause_id}: “{f.quote}”")
            cited.append(f.clause_id)
        return Decision(Outcome.REJECTED, reasons, cited, {"stage": "exclusion"})

    partial = [f for f in exclusions if f.scope == Scope.SOME_PARTS]
    covered = [p for p in damaged_parts
               if not any(p in f.parts for f in partial)]

    if partial and not covered:
        for f in partial:
            reasons.append(f"every damaged part is excluded by {f.clause_id}: "
                           f"“{f.quote}”")
            cited.append(f.clause_id)
        return Decision(Outcome.REJECTED, reasons, cited, {"stage": "exclusion"})

    waivers = {a.lower() for a in (schedule.get("addons") or [])} & DEPRECIATION_WAIVERS
    if waivers and limits:
        # Every limit Section I of the standard form imposes is a depreciation rate -
        # by material, by age, or on paint - so a depreciation add-on waives all of
        # them. Keying off the quote text instead missed limits whose quote happened
        # not to contain the word or a percentage.
        reasons.append(f"depreciation is not deducted: the schedule shows "
                       f"{', '.join(sorted(waivers))}")
        limits = []

    estimate = schedule.get("repair_estimate")
    idv = schedule.get("idv_total")
    if estimate and idv and estimate >= CTL_FRACTION * idv:
        return Decision(Outcome.NEEDS_MANUAL_REVIEW,
                        [f"the repair estimate of Rs {estimate:,} is "
                         f"{estimate / idv:.0%} of the IDV of Rs {idv:,}, at or above the "
                         f"{CTL_FRACTION:.0%} constructive total loss threshold, so this "
                         f"is a total loss assessment rather than a repair claim"],
                        [c for c in cited if c], {"stage": "total_loss"})

    for f in limits:
        cited.append(f.clause_id)

    if partial or limits:
        for f in partial:
            reasons.append(f"{', '.join(f.parts)} excluded by {f.clause_id}: "
                           f"“{f.quote}”")
        for f in limits:
            applies = f", applied to {', '.join(f.parts)}" if f.parts else ""
            reasons.append(f"payment reduced by {f.clause_id}{applies}: “{f.quote}”")
        if schedule.get("compulsory_deductible"):
            reasons.append(f"compulsory deductible of Rs "
                           f"{schedule['compulsory_deductible']:,} applies")
        return Decision(Outcome.PARTIALLY_APPROVED, reasons,
                        [c for c in cited if c],
                        {"stage": "limits", "covered_parts": covered})

    reasons.append(f"the cause is insured and no exclusion applies to "
                   f"{', '.join(damaged_parts)}")
    if schedule.get("compulsory_deductible"):
        reasons.append(f"compulsory deductible of Rs "
                       f"{schedule['compulsory_deductible']:,} applies")
    return Decision(Outcome.APPROVED, reasons, [c for c in cited if c],
                    {"stage": "approved", "covered_parts": covered})
