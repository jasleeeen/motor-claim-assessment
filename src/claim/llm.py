"""Read the policy clauses with Groq and report findings.
The model's only job is to say what the clauses say and point at where. It does not
decide the claim; src/claim/decision.py does that from validated findings.
"""
from __future__ import annotations

import json
import os
import re
import time

from src.claim.decision import Finding, Scope, validate_citation

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"


def endpoint() -> str:
    base = os.environ.get("LLM_BASE_URL")
    if not base:
        return GROQ_URL
    base = base.rstrip("/")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def model_name() -> str:
    return os.environ.get("LLM_MODEL") or MODEL


def api_key_for(explicit: str | None = None) -> str | None:
    return explicit or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
TIMEOUT = 90
RETRY_ON_RATE_LIMIT = 6
MAX_RATE_LIMIT_WAIT = 90.0     
DEFAULT_RATE_LIMIT_WAIT = 25.0 


def retry_after(headers, body: str) -> float:
    """How long to wait after a 429. """
    waits = []
    header = headers.get("retry-after")
    if header:
        try:
            waits.append(float(header))
        except ValueError:
            pass                   
    hint = re.search(r"try again in\s+([0-9.]+)\s*(ms|s)\b", body, re.I)
    if hint:
        seconds = float(hint.group(1))
        waits.append(seconds / 1000 if hint.group(2).lower() == "ms" else seconds)
    if not waits:
        return DEFAULT_RATE_LIMIT_WAIT
    return min(max(waits) + 1.0, MAX_RATE_LIMIT_WAIT)


def daily_limit(body: str) -> bool:
    """True when the 429 is a per-day cap rather than a per-minute one."""
    return bool(re.search(r"per\s*day|\bTPD\b|\bRPD\b|tokens per day", body, re.I))

SYSTEM_PROMPT = """You read motor insurance policy clauses. You do not decide claims.

You are given the clauses of one policy, each with an id in square brackets, plus the
facts of one incident and the vehicle parts that were damaged.

Report only what the given clauses say:

1. cause_insured - does a clause in SECTION_I list the stated cause as an insured
   peril? Set "insured" true with the clause id that lists it, or false with the clause
   id of the cover grant that omits it.
2. exclusions - clauses that remove cover for this incident. scope "whole_claim" if the
   clause defeats the entire claim, or "some_parts" with the affected part names if it
   removes only particular parts.
3. limits - clauses that reduce the amount payable without removing cover, such as a
   depreciation rate on certain materials.

Rules:
- Use ONLY the clauses given. Never rely on insurance knowledge from elsewhere.
- Every finding must cite a clause id exactly as printed in brackets.
- Every quote must be copied word for word from that clause, 25 words or fewer.
- If no exclusion applies, return an empty list. Do not invent one.
- Judge the cause from the incident facts alone. Do not assume fault, drink driving or
  an invalid licence unless the facts say so."""

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["cause_insured", "exclusions", "limits"],
    "properties": {
        "cause_insured": {
            "type": "object", "additionalProperties": False,
            "required": ["insured", "clause_id", "quote"],
            "properties": {
                "insured": {"type": "boolean"},
                "clause_id": {"type": "string"},
                "quote": {"type": "string"},
            },
        },
        "exclusions": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["clause_id", "quote", "scope", "parts"],
                "properties": {
                    "clause_id": {"type": "string"},
                    "quote": {"type": "string"},
                    "scope": {"type": "string", "enum": ["whole_claim", "some_parts"]},
                    "parts": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "limits": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["clause_id", "quote", "parts"],
                "properties": {
                    "clause_id": {"type": "string"},
                    "quote": {"type": "string"},
                    "parts": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


class GroqUnavailable(RuntimeError):
    """No API key, or Groq could not be reached."""


def build_user_message(incident: dict, damaged_parts: list[str], clause_text: str) -> str:
    return (f"INCIDENT\n{json.dumps(incident, indent=1, default=str)}\n\n"
            f"DAMAGED PARTS\n{json.dumps(damaged_parts)}\n\n"
            f"POLICY CLAUSES\n{clause_text}")


def call_groq(messages: list[dict], *, api_key: str | None = None,
              use_schema: bool = True) -> str:
    """One chat completion. Retries only on rate limiting, which the free tier hits."""
    import httpx

    api_key = api_key_for(api_key)
    if not api_key:
        raise GroqUnavailable("GROQ_API_KEY is not set")

    payload = {"model": model_name(), "messages": messages, "temperature": 0}
    if use_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "policy_findings", "strict": True,
                            "schema": RESPONSE_SCHEMA},
        }
    else:
        payload["response_format"] = {"type": "json_object"}

    last = ""
    for attempt in range(RETRY_ON_RATE_LIMIT + 1):
        try:
            r = httpx.post(endpoint(), headers={"Authorization": f"Bearer {api_key}"},
                           json=payload, timeout=TIMEOUT)
        except Exception as exc:                      # network, DNS, proxy
            raise GroqUnavailable(
                f"could not reach {endpoint()}: {exc}") from exc
        if r.status_code == 429:
            body = r.text[:500]
            if daily_limit(body):
                raise GroqUnavailable(
                    "Groq daily token limit reached, so waiting will not help today. "
                    "The keyword baseline needs no key: "
                    "python -m scripts.evaluate_claims")
            wait = retry_after(r.headers, body)
            last = f"waited {wait:.0f}s"
            if attempt == RETRY_ON_RATE_LIMIT:
                break
            print(f"    rate limited, waiting {wait:.0f}s "
                  f"(retry {attempt + 1}/{RETRY_ON_RATE_LIMIT})", flush=True)
            time.sleep(wait)
            continue
        if r.status_code >= 400:
            raise GroqUnavailable(
                f"{endpoint()} returned {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"]
    raise GroqUnavailable(
        f"still rate limited after {RETRY_ON_RATE_LIMIT} retries ({last}). "
        f"The free tier allows 8,000 tokens a minute and each claim costs about "
        f"2,500, so a run started while an earlier one is still inside the window "
        f"will hit this. Wait a full minute, then start again - the run has no "
        f"resume, so it restarts from C01.")


def parse_findings(raw: str, clauses: dict[str, str]) -> tuple[Finding, list[Finding], list[Finding]]:
    """Turn the reply into validated findings. Never trusts a citation."""
    data = json.loads(raw)

    c = data["cause_insured"]
    cause = validate_citation(
        Finding(clause_id=c.get("clause_id"), quote=c.get("quote"),
                detail="insured" if c.get("insured") else "not_insured"), clauses)

    exclusions = [
        validate_citation(
            Finding(clause_id=e.get("clause_id"), quote=e.get("quote"),
                    scope=Scope(e.get("scope", "whole_claim")),
                    parts=list(e.get("parts") or [])), clauses)
        for e in data.get("exclusions", [])]

    limits = [
        validate_citation(
            Finding(clause_id=l.get("clause_id"), quote=l.get("quote"),
                    scope=Scope.SOME_PARTS, parts=list(l.get("parts") or [])), clauses)
        for l in data.get("limits", [])]

    return cause, exclusions, limits


def read_policy(incident: dict, damaged_parts: list[str], clauses: dict[str, str],
                clause_text: str, *, api_key: str | None = None, retries: int = 1,
                **_):
    """Ask the model, validate, and give it one chance to correct bad citations."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": build_user_message(incident, damaged_parts, clause_text)}]

    for attempt in range(retries + 1):
        raw = call_groq(messages, api_key=api_key)
        cause, exclusions, limits = parse_findings(raw, clauses)
        bad = [f for f in [cause, *exclusions, *limits] if not f.citation_valid]
        if not bad:
            return cause, exclusions, limits, raw
        if attempt == retries:
            return cause, exclusions, limits, raw
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user",
             "content": "These citations did not check out: "
                        + "; ".join(f"{f.clause_id}: {f.citation_problem}" for f in bad)
                        + ". Correct them using only the clause ids printed in brackets "
                          "above, and quote the clause word for word."},
        ]
    raise AssertionError("unreachable")
