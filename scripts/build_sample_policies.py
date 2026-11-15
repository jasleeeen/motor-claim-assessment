"""Build the sample policies: a synthetic schedule bound to a real insurer wording.
Each case exercises one branch of the claim-level checks, and the four wordings are
rotated across them so every document layout gets parsed.

    python -m scripts.build_sample_policies
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.policy.schedule import Schedule, bind_schedule_to_wording, write_schedule_pdf
from src.paths import RAW_POLICIES, SAMPLE_POLICIES

RAW = RAW_POLICIES
OUT = SAMPLE_POLICIES

BASE = dict(
    insurer="Specimen General Insurance Ltd.",
    product_name="Private Car Package Policy",
    uin="IRDAN000RPMT0001V01202425",
    insured_address="12 Example Road, Pune 411001",
    make_model="Maruti Suzuki Swift VXI",
    year_of_manufacture=2022,
    engine_number="K12MN1234567",
    chassis_number="MA3EJKD1S00123456",
    fuel="Petrol",
    idv_vehicle=450_000,
    idv_electrical=15_000,
)

CASES = [
    ("active-package", "digit",
     "Normal in-force package policy. An own-damage claim should proceed to the clauses.",
     dict(cover_type="PACKAGE", period_from=date(2026, 4, 1), period_to=date(2027, 3, 31),
          insured_name="A. Sample", registration_number="MH12AB1234", ncb_percent=25,
          addons=["Tyre Protect"])),

    ("expired", "bajaj",
     "Cover ended before any 2026 incident. Must REJECT on the policy period alone.",
     dict(cover_type="PACKAGE", period_from=date(2024, 4, 1), period_to=date(2025, 3, 31),
          insured_name="B. Lapsed", registration_number="MH12CD5678", ncb_percent=0)),

    ("not-yet-started", "digit",
     "Cover begins in the future. Must REJECT: no cover on the incident date.",
     dict(cover_type="PACKAGE", period_from=date(2027, 6, 1), period_to=date(2028, 5, 31),
          insured_name="C. Early", registration_number="MH12EF9012", ncb_percent=0)),

    ("liability-only", "tataaig",
     "Third-party cover only, no Section I. Must REJECT any own-damage claim.",
     dict(cover_type="LIABILITY_ONLY", period_from=date(2026, 4, 1), period_to=date(2027, 3, 31),
          insured_name="D. ThirdParty", registration_number="MH12GH3456",
          make_model="Tata Nexon XZ",
          idv_vehicle=0, idv_electrical=0, ncb_percent=0)),

    ("standalone-od", "icici",
     "Own-damage only. Vehicle damage is covered; a third-party claim would not be.",
     dict(cover_type="STANDALONE_OD", period_from=date(2026, 4, 1), period_to=date(2027, 3, 31),
          insured_name="E. OwnDamage", registration_number="MH12IJ7890", ncb_percent=20)),

    ("high-voluntary-deductible", "bajaj",
     "Rs 7,500 voluntary excess on top of the compulsory deductible.",
     dict(cover_type="PACKAGE", period_from=date(2026, 4, 1), period_to=date(2027, 3, 31),
          insured_name="F. Excess", registration_number="MH12KL2345",
          voluntary_deductible=7_500, ncb_percent=35)),

    ("zero-depreciation", "icici",
     "Parts Depreciation Protect bought, so the depreciation schedule should not bite.",
     dict(cover_type="PACKAGE", period_from=date(2026, 4, 1), period_to=date(2027, 3, 31),
          insured_name="G. ZeroDep", registration_number="MH12MN6789", ncb_percent=45,
          addons=["Parts Depreciation Protect", "Consumable Cover", "Return To Invoice"])),

    ("old-vehicle-high-depreciation", "tataaig",
     "Nine-year-old car, so the highest depreciation band and a low IDV apply.",
     dict(cover_type="PACKAGE", period_from=date(2026, 4, 1), period_to=date(2027, 3, 31),
          insured_name="H. Vintage", registration_number="MH12OP0123",
          make_model="Hyundai i20 Magna", year_of_manufacture=2017,
          idv_vehicle=120_000, idv_electrical=0, ncb_percent=50)),
]


def build() -> list[dict]:
    missing = sorted({w for _, w, _, _ in CASES if not (RAW / f"{w}.pdf").exists()})
    if missing:
        raise SystemExit(
            "The insurer wordings are not in data/raw_policies/.\n\n"
            f"  missing: {', '.join(n + '.pdf' for n in missing)}\n\n"
            "They are published documents but are not redistributed with this\n"
            'repository. See "Getting the insurer wordings" in README.md for the\n'
            "filenames and the regulator UIN that identifies each one.\n\n"
            "The eight built policies are already in data/sample_policies/, so\n"
            "nothing needs rebuilding unless you are changing the schedules.")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for n, (case_id, wording, why, overrides) in enumerate(CASES, start=1):
        s = Schedule(policy_number=f"OG-26-0000-1801-{n:08d}", **{**BASE, **overrides})
        sched_pdf = write_schedule_pdf(s, OUT / f"_schedule_{case_id}.pdf")
        bound = bind_schedule_to_wording(sched_pdf, RAW / f"{wording}.pdf",
                                         OUT / f"{case_id}.pdf")
        sched_pdf.unlink()
        manifest.append({"case": case_id, "file": bound.name, "wording": wording,
                         "why": why, "schedule": s.as_dict()})
        print(f"  {case_id:30s} <- {wording:8s} {s.cover_type:14s} "
              f"{s.period_from}..{s.period_to}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(manifest)} sample policies -> {OUT}/  (+ manifest.json)")
    return manifest


if __name__ == "__main__":
    build()
