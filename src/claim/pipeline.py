"""Policy PDF + incident + damaged parts -> a decision with citations."""
from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path

from src.claim import baseline
from src.claim.decision import Decision, Outcome, decide
from src.policy.clauses import as_lookup, build_index, render_for_prompt
from src.policy.extract import extract_lines
from src.policy.integrity import check_policy
from src.policy.schedule import ScheduleNotFound, parse_schedule
from src.policy.sections import select_relevant
from src.vision.linking import VisionResult

READERS = {"baseline": baseline.read_policy}


def _llm_reader():
    from src.claim import llm
    return llm.read_policy


def load_policy(policy_pdf: str | Path) -> dict:
    """Everything the decision needs from the PDF, read once and cacheable."""
    lines = extract_lines(policy_pdf)
    sections = select_relevant(lines)
    clauses = build_index(lines, sections)
    try:
        schedule = parse_schedule(lines)
    except ScheduleNotFound as exc:
        schedule = {"error": str(exc)}
    lookup = as_lookup(clauses)
    return {"schedule": schedule, "clauses": clauses,
            "lookup": lookup, "prompt_text": render_for_prompt(clauses),
            "sections": [s.kind for s in sections],
            "integrity": check_policy(lookup)}


def assess_claim(policy_pdf: str | Path, incident: dict,
                 damaged_parts: list[str] | None = None,
                 *, reader: str = "baseline", api_key: str | None = None,
                 vision: VisionResult | None = None,
                 vision_needs_review: bool = False,
                 vision_review_reasons: list[str] | None = None,
                 policy: dict | None = None, claim_id: str | None = None) -> dict:
    """damaged_parts may be given directly, or taken from a VisionResult.

    When both are given the explicit list wins, so a human can correct what the
    detector read without losing its review flags.
    """
    if vision is not None:
        if damaged_parts is None:
            damaged_parts = vision.part_names()
        vision_needs_review = vision_needs_review or vision.needs_review
        vision_review_reasons = list(vision_review_reasons or []) + vision.review_reasons
    damaged_parts = list(damaged_parts or [])
    policy = policy or load_policy(policy_pdf)
    schedule = policy["schedule"]

    if "error" in schedule:
        return _as_dict(Decision(Outcome.NEEDS_MANUAL_REVIEW,
                                 [f"the policy document has no readable schedule: "
                                  f"{schedule['error']}"]),
                        policy, incident, damaged_parts, reader, None)

    problems = policy.get("integrity") or []
    if problems:
        return _as_dict(Decision(Outcome.NEEDS_MANUAL_REVIEW, problems,
                                 detail={"stage": "policy_integrity"}),
                        policy, incident, damaged_parts, reader, None)

    # A claim form with no stated cause must not be rejected outright
    if not incident.get("cause"):
        return _as_dict(Decision(Outcome.NEEDS_MANUAL_REVIEW,
                                 ["the claim does not say what caused the damage"],
                                 detail={"stage": "incident"}),
                        policy, incident, damaged_parts, reader, None)

    incident_date = incident.get("date")
    if isinstance(incident_date, str):
        incident_date = date.fromisoformat(incident_date)
    if incident_date is None:
        return _as_dict(Decision(Outcome.NEEDS_MANUAL_REVIEW,
                                 ["the claim does not state when the incident happened"]),
                        policy, incident, damaged_parts, reader, None)

    read = READERS.get(reader) or _llm_reader()
    cause, exclusions, limits, raw = read(
        incident, damaged_parts, policy["lookup"],
        clause_text=policy["prompt_text"], api_key=api_key, claim_id=claim_id)

    decision = decide(schedule, incident_date, damaged_parts, cause, exclusions, limits,
                      vision_needs_review=vision_needs_review,
                      vision_review_reasons=vision_review_reasons)
    return _as_dict(decision, policy, incident, damaged_parts, reader, raw,
                    cause, exclusions, limits)


def _as_dict(decision, policy, incident, damaged_parts, reader, raw,
             cause=None, exclusions=None, limits=None) -> dict:
    schedule = policy.get("schedule", {})
    return {
        "outcome": decision.outcome.value,
        "reasons": decision.reasons,
        "clauses_cited": {cid: policy["lookup"].get(cid, "") for cid in decision.cited},
        "damaged_parts": damaged_parts,
        "incident": incident,
        "policy": {k: str(schedule.get(k)) for k in
                   ("policy_number", "cover_type", "period_from", "period_to")},
        "findings": {
            "cause_insured": asdict(cause) if cause else None,
            "exclusions": [asdict(f) for f in (exclusions or [])],
            "limits": [asdict(f) for f in (limits or [])],
        },
        "reader": reader,
        "clauses_available": len(policy.get("lookup", {})),
        "stage": decision.detail.get("stage"),
    }
