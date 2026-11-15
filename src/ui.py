from __future__ import annotations

import os
from pathlib import Path
from src.paths import WEIGHTS

OUTCOME_STYLE = {"APPROVED": "success", "PARTIALLY_APPROVED": "warning",
                 "NEEDS_MANUAL_REVIEW": "warning", "REJECTED": "error"}
OUTCOME_BLURB = {
    "APPROVED": "The cause is insured and nothing in the policy reduces or removes cover.",
    "PARTIALLY_APPROVED": "Cover applies, but the policy reduces what is payable.",
    "REJECTED": "The policy does not cover this claim.",
    "NEEDS_MANUAL_REVIEW": "This cannot be settled automatically and needs a human.",
}
# The causes Section I of the standard form lists, plus an escape hatch.
CAUSES = ["accident", "fire", "theft", "riot", "earthquake", "flood", "malicious",
          "terrorism", "transit", "landslide", "wear", "mechanical", "other"]
EXCLUSION_FLAGS = {
    "intoxicated": "Driver was under the influence",
    "no_valid_licence": "Driver held no valid licence",
    "outside_geographical_area": "Incident outside the geographical area",
    "commercial_use": "Private car used commercially",
}
DEFAULT_WEIGHTS = WEIGHTS


def parse_parts(text: str) -> list[str]:
    """Comma-separated part names, trimmed, duplicates removed, order kept."""
    seen, out = set(), []
    for raw in (text or "").split(","):
        name = " ".join(raw.split())
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out


def decision_rows(result: dict) -> list[dict]:
    """One row per clause the decision relied on."""
    return [{"Clause": cid, "What it says": text}
            for cid, text in (result.get("clauses_cited") or {}).items()]


def finding_rows(result: dict) -> list[dict]:
    """Everything the reader reported, including citations that failed validation."""
    findings = result.get("findings") or {}
    rows = []
    cause = findings.get("cause_insured")
    if cause:
        rows.append({"Kind": "cause",
                     "Verdict": "insured" if cause.get("detail") == "insured" else "not insured",
                     "Clause": cause.get("clause_id") or "-",
                     "Quoted": cause.get("quote") or "-",
                     "Citation": _citation(cause)})
    for f in findings.get("exclusions") or []:
        rows.append({"Kind": f"exclusion ({f.get('scope')})", "Verdict": "applies",
                     "Clause": f.get("clause_id") or "-", "Quoted": f.get("quote") or "-",
                     "Citation": _citation(f)})
    for f in findings.get("limits") or []:
        rows.append({"Kind": "limit", "Verdict": ", ".join(f.get("parts") or []) or "whole claim",
                     "Clause": f.get("clause_id") or "-", "Quoted": f.get("quote") or "-",
                     "Citation": _citation(f)})
    return rows


def _citation(f: dict) -> str:
    return "verified" if f.get("citation_valid") else f"REJECTED - {f.get('citation_problem')}"


def build_incident(date_value, cause: str, description: str, flags: dict) -> dict:
    incident = {"date": date_value.isoformat() if date_value else None,
                "cause": cause or None, "description": description or ""}
    incident.update({k: True for k, v in flags.items() if v})
    return incident


def available_readers(weights_dir: Path = DEFAULT_WEIGHTS) -> dict:
    """What this installation can actually do, checked rather than assumed."""
    try:
        import torch  # noqa: F401
        has_torch = True
    except (ImportError, OSError):
        # OSError: installed, but the OS refused to load its DLLs.
        has_torch = False
    damage = Path(weights_dir) / "damage.pt"
    parts = Path(weights_dir) / "parts.pt"
    from src.claim.llm import api_key_for

    return {"torch": has_torch,
            "damage_weights": damage.exists(), "parts_weights": parts.exists(),
            "vision": has_torch and damage.exists() and parts.exists(),
            # LLM_API_KEY or GROQ_API_KEY: the reader works with any
            # OpenAI-compatible provider, not only Groq.
            "groq_key": bool(api_key_for()),
            "damage_path": damage, "parts_path": parts}


TOKENS_PER_CLAIM = 3_500

def reader_choices(caps: dict) -> list[str]:
    """Readers this installation can actually run, cheapest first."""
    out = ["baseline"]
    if caps.get("groq_key"):
        out.append("llm")
    return out


READER_LABELS = {
    "baseline": "keyword baseline (free, no tokens)",
    "llm": "language model (Groq or compatible)",
}


def markdown_table(rows: list[dict]) -> str: 
    if not rows:
        return ""
    headers = list(rows[0])

    def cell(value: object) -> str:
        text = "" if value is None else str(value)
        text = text.replace("\\", "\\\\").replace("|", "\\|")
        # A newline inside a cell breaks the row; collapse all whitespace runs.
        return " ".join(text.split()) or "-"

    lines = ["| " + " | ".join(cell(h) for h in headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(cell(r.get(h)) for h in headers) + " |" for r in rows]
    return "\n".join(lines)