"""Run the labelled claims through Groq and compare against the keyword baseline.
Needs GROQ_API_KEY in .env. Paced for the free tier, so about 19 minutes for 28.

    python -m scripts.run_groq_eval
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from scripts.evaluate_claims import main as evaluate
from src.paths import OUTPUTS
from src.env import load_env

PAUSE_SECONDS = 40


def main(pause: float = PAUSE_SECONDS, limit: int | None = None,
         resume: bool = True) -> dict:
    import src.claim.pipeline as pipeline
    from src.claim import llm

    original = llm.read_policy

    def paced(*args, **kwargs):
        out = original(*args, **kwargs)
        time.sleep(pause)
        return out

    llm.read_policy = paced
    pipeline.READERS.pop("llm", None)
    try:
        groq = evaluate("llm", limit=limit, resume=resume)
    finally:
        llm.read_policy = original

    _base_file = OUTPUTS / "eval_baseline.json"
    base = json.loads(_base_file.read_text()) \
        if _base_file.exists() else evaluate("baseline", verbose=False)

    print(f"\n  baseline {base['correct']}/{base['n']} ({base['accuracy']:.0%})"
          f"    groq {groq['correct']}/{groq['n']} ({groq['accuracy']:.0%})")
    by_id = {r["id"]: r for r in base["rows"]}
    print("\n  where they differ:")
    for r in groq["rows"]:
        b = by_id.get(r["id"])
        if b and b["got"] != r["got"]:
            print(f"    {r['id']} want {r['expected']:20s} baseline {b['got']:20s} groq {r['got']}")
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "eval_comparison.json").write_text(json.dumps(
        {"baseline": base["accuracy"], "groq": groq["accuracy"],
         "n": groq["n"],
         "differences": [{"id": r["id"], "expected": r["expected"],
                          "baseline": by_id[r["id"]]["got"], "groq": r["got"]}
                         for r in groq["rows"]
                         if r["id"] in by_id and by_id[r["id"]]["got"] != r["got"]]},
        indent=1))
    return groq


def _cli() -> int:
    """A missing key is the most likely way this is run wrong, and it is not a bug.
    A raw traceback for a foreseeable mistake reads as broken code, so say what to
    do instead."""
    import os

    from src.claim.llm import GroqUnavailable

    load_env()         
    import argparse

    ap = argparse.ArgumentParser(prog="python -m scripts.run_groq_eval")
    ap.add_argument("--limit", type=int,
                    help="score at most N not-yet-done claims. The free tier allows "
                         "200,000 tokens a day and a full run costs about 98,000, so "
                         "--limit lets a run fit what is left of today's budget")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore saved progress and start from C01")
    args = ap.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set, so the policy reader cannot be called.\n\n"
              "  PowerShell:  $env:GROQ_API_KEY = \"gsk_...\"\n"
              "  bash:        export GROQ_API_KEY=gsk_...\n\n"
              "A free key comes from https://console.groq.com. In VS Code, put it in\n"
              ".env instead and use the Groq launch configuration.\n\n"
              "For a run with no key at all: python -m scripts.evaluate_claims")
        return 1
    try:
        main(limit=args.limit, resume=not args.fresh)
    except GroqUnavailable as exc:
        print(f"Groq could not be reached: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
