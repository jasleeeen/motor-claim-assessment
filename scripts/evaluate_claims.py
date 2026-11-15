"""Score a reader against the labelled claims in data/test_claims.json.
The labels are hand-written from the policy text, so they are the weakest part of
the evaluation.

    python -m scripts.evaluate_claims
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.claim.pipeline import assess_claim, load_policy
from src.paths import OUTPUTS, SAMPLE_POLICIES, TEST_CLAIMS
from src.env import load_env

CLAIMS = TEST_CLAIMS
POLICIES = SAMPLE_POLICIES


def partial_path(reader: str) -> Path:
    return OUTPUTS / f"partial_{reader}.json"


def main(reader: str = "baseline", api_key: str | None = None, verbose: bool = True,
         resume: bool = True, limit: int | None = None) -> dict:
    """Score every labelled claim with one reader """
    load_env()
    claims = json.loads(CLAIMS.read_text())
    cache: dict[str, dict] = {}
    confusion = Counter()

    OUTPUTS.mkdir(exist_ok=True)
    done: dict[str, dict] = {}
    if resume and partial_path(reader).exists():
        done = json.loads(partial_path(reader).read_text())
        if done and verbose:
            print(f"  resuming: {len(done)} of {len(claims)} already scored")

    todo = [c for c in claims if c["id"] not in done]
    if limit is not None:
        todo = todo[:limit]

    try:
        for c in todo:
            pdf = POLICIES / f"{c['policy']}.pdf"
            cache.setdefault(c["policy"], load_policy(pdf))
            got = assess_claim(pdf, c["incident"], c["damaged_parts"],
                               reader=reader, api_key=api_key,
                               policy=cache[c["policy"]], claim_id=c["id"])
            done[c["id"]] = {
                "id": c["id"], "policy": c["policy"],
                "expected": c["expected_outcome"], "got": got["outcome"],
                "correct": got["outcome"] == c["expected_outcome"],
                "stage": got["stage"], "reasons": got["reasons"],
                "cited": list(got["clauses_cited"])}
            # Save after every claim so a stopped run can resume.
            partial_path(reader).write_text(json.dumps(done, indent=1))
    except (KeyboardInterrupt, Exception):
        partial_path(reader).write_text(json.dumps(done, indent=1))
        if verbose:
            print(f"\n  stopped after {len(done)}/{len(claims)} claims. "
                  f"Progress saved; run again to continue.")
        raise

    order = {c["id"]: i for i, c in enumerate(claims)}
    rows = sorted(done.values(), key=lambda r: order.get(r["id"], 0))
    if not rows:
        raise SystemExit("no claims scored")
    for r in rows:
        confusion[(r["expected"], r["got"])] += 1

    n_ok = sum(r["correct"] for r in rows)
    if verbose:
        for r in rows:
            mark = "ok  " if r["correct"] else "WRONG"
            print(f"  {mark} {r['id']} {r['policy']:28s} {r['expected']:20s} -> {r['got']}")
            if not r["correct"] and r["reasons"]:
                print(f"         {r['reasons'][0][:100]}")
        scope = "" if len(rows) == len(claims) else f" of {len(claims)} labelled"
        print(f"\n  {reader}: {n_ok}/{len(rows)} correct{scope} "
              f"({n_ok / len(rows):.0%})")
        if len(rows) < len(claims):
            print(f"  PARTIAL: {len(claims) - len(rows)} claims not yet scored. "
                  f"Run again to continue.")
        print("\n  confusion (expected -> got):")
        for (want, got), n in sorted(confusion.items()):
            flag = "" if want == got else "   <-- "
            print(f"    {want:20s} -> {got:20s} {n}{flag}")

    result = {"reader": reader, "n": len(rows), "of_labelled": len(claims),
              "complete": len(rows) == len(claims), "correct": n_ok,
              "accuracy": n_ok / len(rows), "rows": rows,
              "confusion": {f"{a}->{b}": n for (a, b), n in confusion.items()}}
    (OUTPUTS / f"eval_{reader}.json").write_text(json.dumps(result, indent=1))
    return result


def _cli(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="python -m scripts.evaluate_claims")
    p.add_argument("--reader", default="baseline",
                   help="baseline (no key) or llm")
    p.add_argument("--limit", type=int,
                   help="score at most N not-yet-done claims, for a tight token budget")
    p.add_argument("--fresh", action="store_true",
                   help="ignore saved progress and score every claim again")
    a = p.parse_args(argv)
    main(reader=a.reader, limit=a.limit, resume=not a.fresh)


if __name__ == "__main__":
    _cli()
