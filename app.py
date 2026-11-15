"""
Motor claim assessment
Upload photographs and a policy, get a claim decision.
"""
from __future__ import annotations
import json
import tempfile
from datetime import date
from pathlib import Path
import streamlit as st

from src.claim.pipeline import assess_claim, load_policy
from src.env import load_env
from src.ui import (CAUSES, EXCLUSION_FLAGS, OUTCOME_BLURB, OUTCOME_STYLE,
                    READER_LABELS, TOKENS_PER_CLAIM, available_readers,
                    build_incident, decision_rows, finding_rows,
                    markdown_table, parse_parts, reader_choices)

load_env()

st.set_page_config(page_title="Motor Claim Assessment", layout="wide")
st.title("Motor claim assessment")
st.caption("Input photographs and a policy document; get a claim decision with the clauses.")

caps = available_readers()

with st.sidebar:
    st.write("Photo analysis:", "ready" if caps["vision"] else "**off**")
    if not caps["vision"]:
        bits = []
        if not caps["torch"]:
            bits.append("torch is not installed")
        if not caps["damage_weights"]:
            bits.append(f"missing {caps['damage_path']}")
        if not caps["parts_weights"]:
            bits.append(f"missing {caps['parts_path']}")
        st.caption("; ".join(bits) + ". Enter the damaged parts by hand instead.")
    choices = reader_choices(caps)
    reader = st.radio("Policy reader", choices,
                      index=len(choices) - 1 if len(choices) > 1 else 0,
                      format_func=lambda r: READER_LABELS.get(r, r))
    if reader == "baseline":
        if len(choices) == 1:
            st.caption("Set GROQ_API_KEY to use the language model. The **keyword "
                       "baseline** cannot follow cross-references between clauses.")
        else:
            st.caption("Using the **keyword baseline**: no tokens spent. It cannot "
                       "follow cross-references between clauses.")
    else:
        st.caption(f"About {TOKENS_PER_CLAIM:,} tokens per claim. ")
    st.divider()


left, right = st.columns([1, 1])

with left:
    st.subheader("The claim")
    policy_file = st.file_uploader("Policy document (PDF)", type=["pdf"])
    photos = st.file_uploader("Damage photographs", type=["jpg", "jpeg", "png"],
                              accept_multiple_files=True)
    incident_date = st.date_input("Date of the incident", value=date(2026, 11, 15))
    cause = st.selectbox("Cause of the damage", CAUSES, index=0)
    description = st.text_area("What happened", height=80)
    with st.expander("Circumstances that may exclude the claim"):
        flags = {k: st.checkbox(label, key=k) for k, label in EXCLUSION_FLAGS.items()}
    manual_parts = st.text_input(
        "Damaged parts (comma separated)",
        placeholder="Front-bumper, Hood (Leave blank to use the photographs, if the models are installed.)",
        help="Leave blank to use the photographs, if the models are installed.")
    go = st.button("Assess the claim", type="primary")

with right:
    if photos:
        st.subheader("Photographs")
        cols = st.columns(min(len(photos), 3))
        for i, photo in enumerate(photos):
            cols[i % len(cols)].image(photo, use_container_width=True)

if go:
    if policy_file is None:
        st.error("Upload the policy document.")
        st.stop()

    vision = None
    if photos and caps["vision"] and not parse_parts(manual_parts):
        from PIL import Image
        from src.vision.detector import ClaimVision, VisionUnavailable
        try:
            with st.spinner("Reading the photographs..."):
                cv = ClaimVision(caps["damage_path"], caps["parts_path"])
                vision = cv.assess_photos([Image.open(p) for p in photos])
        except VisionUnavailable as exc:
            st.warning(f"Photo analysis unavailable: {exc}")

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(policy_file.getvalue())
            tmp = fh.name
        with st.spinner("Reading the policy..."):
            policy = load_policy(tmp)
            result = assess_claim(
                tmp, build_incident(incident_date, cause, description, flags),
                parse_parts(manual_parts) or None, vision=vision, policy=policy,
                reader=reader)
    except Exception as exc:
        st.error(f"The claim could not be assessed: {exc}")
        st.stop()
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)

    st.session_state["result"] = result
    st.session_state["vision"] = vision

result = st.session_state.get("result")
if result:
    st.divider()
    getattr(st, OUTCOME_STYLE[result["outcome"]])(f"**{result['outcome'].replace('_', ' ').title()}** "
                                                  f"- {OUTCOME_BLURB[result['outcome']]}")
    for reason in result["reasons"]:
        st.write("-", reason)

    vision = st.session_state.get("vision")
    if vision:
        st.subheader("What the photographs showed")
        if vision.damaged_parts:
            st.markdown(markdown_table([{"Part": p, "Damage": ", ".join(k)}
                                        for p, k in vision.damaged_parts.items()]))
        else:
            st.write("No damage was linked to a part.")
        st.caption(f"{len(vision.parts_detected)} parts identified at a confidence "
                   f"threshold of {vision.score_threshold}.")

    st.subheader(f"Damaged parts assessed: {', '.join(result['damaged_parts']) or 'none'}")

    rows = decision_rows(result)
    if rows:
        st.subheader("Clauses the decision rests on")
        for row in rows:
            with st.expander(row["Clause"]):
                st.write(row["What it says"])

    with st.expander("Everything the policy reader reported"):
        st.caption("Including citations that failed validation. A citation is only "
                   "accepted if the clause exists and the quoted words are really in it.")
        st.markdown(markdown_table(finding_rows(result)) or "The reader reported nothing.")

    with st.expander("The policy"):
        st.json(result["policy"])
        st.caption(f"{result['clauses_available']} clauses were extracted and sent to "
                   f"the reader. Reader: {result['reader']}.")

    st.download_button("Download the assessment (JSON)",
                       json.dumps(result, indent=2, ensure_ascii=False),
                       file_name="claim_assessment.json", mime="application/json")
