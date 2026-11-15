"""Generate synthetic policy schedules and bind them to a real insurer wording.
This exists because a published wording is rules without a customer. Every one of the
four sample wordings says "as specified in the Schedule" for the policy period, the
IDV, the deductibles and the geographical area, and leaves them blank. Without those
values there is nothing to check a claim date or a cover type against.

The field structure follows the IRDAI Customer Information Sheet, which every insurer publishes as a blank
template, so the layout is authentic and the people in it are invented.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path

# Add-on names as printed in the Digit Customer Information Sheet.
KNOWN_ADDONS = [
    "Consumable Cover", "Parts Depreciation Protect", "Engine and Gear Box Protect",
    "Tyre Protect", "Key & Lock Protect", "Loss to Personal Belongings",
    "Return To Invoice", "NCB Protector",
]
COVER_TYPES = {
    "PACKAGE": "Package Policy (Section I Own Damage + Section II Liability + Section III PA)",
    "LIABILITY_ONLY": "Liability Only Policy (Section II Liability + Section III PA)",
    "STANDALONE_OD": "Standalone Own Damage Policy (Section I Own Damage only)",
}


@dataclass
class Schedule:
    policy_number: str
    insurer: str
    product_name: str
    uin: str
    cover_type: str             
    insured_name: str
    insured_address: str
    period_from: date
    period_to: date
    make_model: str
    registration_number: str
    year_of_manufacture: int
    engine_number: str
    chassis_number: str
    fuel: str
    idv_vehicle: int
    idv_electrical: int = 0
    idv_non_electrical: int = 0
    idv_cng_lpg: int = 0
    compulsory_deductible: int = 1000
    voluntary_deductible: int = 0
    ncb_percent: int = 0
    addons: list[str] = field(default_factory=list)
    geographical_area: str = "India"

    @property
    def idv_total(self) -> int:
        return (self.idv_vehicle + self.idv_electrical
                + self.idv_non_electrical + self.idv_cng_lpg)

    @property
    def cover_description(self) -> str:
        return COVER_TYPES[self.cover_type]

    def covers_own_damage(self) -> bool:
        return self.cover_type in ("PACKAGE", "STANDALONE_OD")

    def is_active_on(self, when: date) -> bool:
        return self.period_from <= when <= self.period_to

    def as_dict(self) -> dict:
        d = asdict(self)
        d["period_from"] = self.period_from.isoformat()
        d["period_to"] = self.period_to.isoformat()
        d["idv_total"] = self.idv_total
        return d


def _rows(s: Schedule) -> list[tuple[str, str]]:
    """Label/value rows, in the order a real schedule prints them."""
    money = lambda n: f"Rs. {n:,}"
    return [
        ("Policy Number", s.policy_number),
        ("Product Name", s.product_name),
        ("Unique Identification Number (UIN)", s.uin),
        ("Type of Cover", s.cover_description),
        ("Name of Insured", s.insured_name),
        ("Address of Insured", s.insured_address),
        ("Period of Insurance - From", f"{s.period_from:%d/%m/%Y} 00:00 hrs"),
        ("Period of Insurance - To", f"{s.period_to:%d/%m/%Y} 23:59 hrs"),
        ("Geographical Area", s.geographical_area),
        ("Make and Model", s.make_model),
        ("Registration Number", s.registration_number),
        ("Year of Manufacture", str(s.year_of_manufacture)),
        ("Engine Number", s.engine_number),
        ("Chassis Number", s.chassis_number),
        ("Fuel Type", s.fuel),
        ("Vehicle IDV", money(s.idv_vehicle)),
        ("Non-electrical Accessories IDV", money(s.idv_non_electrical)),
        ("Electrical Accessories IDV", money(s.idv_electrical)),
        ("CNG/LPG IDV", money(s.idv_cng_lpg)),
        ("Total IDV", money(s.idv_total)),
        ("Compulsory Deductible", money(s.compulsory_deductible)),
        ("Voluntary Deductible", money(s.voluntary_deductible)),
        ("No Claim Bonus", f"{s.ncb_percent}%"),
        ("Add-on Covers", ", ".join(s.addons) if s.addons else "None"),
    ]


def write_schedule_pdf(s: Schedule, out_path: str | Path) -> Path:
    """Render the schedule as a one-page PDF, in a label/value layout."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    cell = ParagraphStyle("cell", parent=styles["BodyText"], fontSize=8.5, leading=11)

    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            topMargin=15 * mm, bottomMargin=15 * mm,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            title=f"Policy Schedule {s.policy_number}")
    story = [
        Paragraph(f"<b>{s.insurer}</b>", styles["Heading2"]),
        Paragraph("POLICY SCHEDULE", styles["Heading3"]),
        Paragraph("<i>Synthetic schedule for testing. The named insured, vehicle and "
                  "identifiers are invented and refer to no real person or vehicle.</i>", cell),
        Spacer(1, 5 * mm),
        Table([[Paragraph(f"<b>{k}</b>", cell), Paragraph(v, cell)] for k, v in _rows(s)],
              colWidths=[62 * mm, 112 * mm],
              style=TableStyle([
                  ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                  ("VALIGN", (0, 0), (-1, -1), "TOP"),
                  ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f2f2")),
                  ("LEFTPADDING", (0, 0), (-1, -1), 4),
                  ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                  ("TOPPADDING", (0, 0), (-1, -1), 3),
                  ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
              ])),
        Spacer(1, 4 * mm),
        Paragraph("Subject to the attached policy wording, which governs all terms, "
                  "exceptions, conditions and limitations.", cell),
    ]
    doc.build(story)
    return out_path


def bind_schedule_to_wording(schedule_pdf: str | Path, wording_pdf: str | Path,
                             out_path: str | Path) -> Path:
    """Schedule first, then the insurer's wording - the order a real policy arrives in."""
    from pypdf import PdfReader, PdfWriter

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for src in (schedule_pdf, wording_pdf):
        for page in PdfReader(str(src)).pages:
            writer.add_page(page)
    with open(out_path, "wb") as fh:
        writer.write(fh)
    return out_path



import re
from datetime import datetime

_LABELS = {
    "Policy Number": "policy_number",
    "Product Name": "product_name",
    "Type of Cover": "cover_description",
    "Name of Insured": "insured_name",
    "Period of Insurance - From": "period_from",
    "Period of Insurance - To": "period_to",
    "Geographical Area": "geographical_area",
    "Make and Model": "make_model",
    "Registration Number": "registration_number",
    "Year of Manufacture": "year_of_manufacture",
    "Fuel Type": "fuel",
    "Vehicle IDV": "idv_vehicle",
    "Total IDV": "idv_total",
    "Compulsory Deductible": "compulsory_deductible",
    "Voluntary Deductible": "voluntary_deductible",
    "No Claim Bonus": "ncb_percent",
    "Add-on Covers": "addons",
}
_MONEY = re.compile(r"Rs\.?\s*([\d,]+)")
_DATE = re.compile(r"(\d{2})/(\d{2})/(\d{4})")


class ScheduleNotFound(ValueError):
    """The PDF has no schedule page, so no claim-level check can be made."""


def parse_schedule(lines) -> dict:
    """Read the schedule fields back from the extracted lines of a bound policy. """
    texts = [l.text for l in lines if l.page == 1]
    if "POLICY SCHEDULE" not in texts:
        raise ScheduleNotFound("no POLICY SCHEDULE heading on page 1")

    found: dict = {}
    for i, text in enumerate(texts[:-1]):
        key = _LABELS.get(text)
        if key and key not in found:
            found[key] = texts[i + 1].strip()

    out: dict = {}
    for key, raw in found.items():
        if key in ("period_from", "period_to"):
            m = _DATE.search(raw)
            out[key] = datetime(int(m[3]), int(m[2]), int(m[1])).date() if m else None
        elif key.startswith("idv_") or key.endswith("deductible"):
            m = _MONEY.search(raw)
            out[key] = int(m[1].replace(",", "")) if m else None
        elif key == "ncb_percent":
            m = re.search(r"(\d+)", raw)
            out[key] = int(m[1]) if m else None
        elif key == "year_of_manufacture":
            out[key] = int(raw) if raw.isdigit() else None
        elif key == "addons":
            out[key] = [] if raw.lower() == "none" else [a.strip() for a in raw.split(",")]
        else:
            out[key] = raw

    desc = out.get("cover_description", "")
    for code, label in COVER_TYPES.items():
        if desc == label:
            out["cover_type"] = code
            break
    else:
        out["cover_type"] = None

    missing = [k for k in ("policy_number", "period_from", "period_to", "cover_type")
               if not out.get(k)]
    if missing:
        raise ScheduleNotFound(f"schedule is missing required fields: {missing}")
    return out


def covers_own_damage(parsed: dict) -> bool:
    return parsed.get("cover_type") in ("PACKAGE", "STANDALONE_OD")


def is_active_on(parsed: dict, when) -> bool:
    return parsed["period_from"] <= when <= parsed["period_to"]
