#!/usr/bin/env python3

"""
Clinical‑Trial Outcome Extractor – **v11.2 (metadata rows)**
───────────────────────────────────────────────────────────
Adds study‑level metadata *as the first block of rows* in the publication‑ready
CSV, per user request.

**New metadata rows (in this order)**
1. Last author surname + publication year
2. Paper title
3. Journal
4. Health‑care setting (hospital, outpatient, etc.)
5. Country where patients were recruited
6. Patient population
7. Targeted condition (plus diagnostic criteria)
8. Intervention tested
9. Comparator / control

The rest of the v11 logic is unchanged.
"""

import os
import json
import re
import io
from typing import List, Tuple

import pdfplumber
import streamlit as st
import pandas as pd
from openai import OpenAI

# ---------------- CONFIG ----------------
MODEL = "gpt-4o"
DEFAULT_TOKENS = 4096
LARGE_TOKENS = 8192
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))

# -------------- HELPERS -----------------

@st.cache_data
def get_pdf_text(file_bytes: bytes):
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return None


def ask_llm(prompt: str, *, json_mode: bool = True, max_tokens: int = DEFAULT_TOKENS):
    """Call OpenAI with mandatory system msg to satisfy json_object rule."""
    msgs = [
        {
            "role": "system",
            "content": "You are a json‑only assistant. Always respond with a single valid JSON document.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"} if json_mode else {"type": "text"},
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return None


def parse_json(text: str, key: str | None):
    if not text:
        return None
    try:
        data = json.loads(text.strip().removeprefix("```json").removesuffix("```"))
        return data if key is None else data.get(key)
    except json.JSONDecodeError:
        st.warning("LLM did not return valid JSON.")
        return None

# ------------ AGENTS -------------

def agent_extract_metadata(txt: str):
    prompt = (
        "You are a metadata extraction specialist. Extract the following fields as JSON:\n"
        "first_author_surname, last_author_surname, publication_year, paper_title, journal, study_design, "
        "health_care_setting, study_country, patient_population, targeted_condition, diagnostic_criteria, "
        "interventions_tested, comparison_group.\n"
        "If a value is absent, use null. Respond as {\"study_info\": {…}}.\n\n"
        f"Text (first 8k chars):\n{txt[:8000]}"
    )
    return parse_json(ask_llm(prompt), "study_info")


def agent_locate_defined_outcomes(txt: str):
    prompt = (
        "You are a clinical‑trial protocol analyst. Extract all outcome definitions from the Methods section.\n\n"
        "Rules: Treat semicolon lists as separate domains; split by timepoints.\n\n"
        "Return JSON {\"defined_outcomes\": […]}.\n\n"
        f"Doc text:\n{txt}"
    )
    return parse_json(ask_llm(prompt), "defined_outcomes") or []


def agent_parse_table(tbl: str):
    prompt = (
        "You are an expert at parsing clinical‑trial tables. Decide if baseline (return []), else strip stats and build domain/specific pairs.\n\n"
        "Return JSON {\"table_outcomes\": […]}.\n\n"
        f"Table text:\n{tbl}"
    )
    return parse_json(ask_llm(prompt), "table_outcomes") or []


def agent_finalize(lst: list):
    prompt = (
        "You are a data‑structuring expert. Clean & deduplicate into hierarchical outcomes.\n"
        "Return JSON {\"final_outcomes\": […]}.\n\n"
        f"Raw list:\n{json.dumps(lst, indent=2)}"
    )
    return parse_json(ask_llm(prompt, max_tokens=LARGE_TOKENS), "final_outcomes") or []

# ----------- PIPELINE ------------

@st.cache_data(show_spinner="🔍 Extracting…")
def run_pipeline(txt: str):
    meta = agent_extract_metadata(txt)
    defined = agent_locate_defined_outcomes(txt)
    tables = re.findall(r"(Table\s+\d+\..*?)(?=\n(?:Table\s+\d+\.|Figure\s+\d+\.|$))", txt, re.DOTALL)
    tbl_outs = []
    for t in tables:
        tbl_outs.extend(agent_parse_table(t))
    raw = defined + tbl_outs
    finals = agent_finalize(raw) if raw else []
    return meta, finals

# ----------- STREAMLIT UI ---------

st.set_page_config(layout="wide")
st.title("Clinical‑Trial Outcome Extractor v11.2 (metadata rows)")

pdf = st.file_uploader("Upload a clinical‑trial PDF", type="pdf")
if pdf:
    txt = get_pdf_text(pdf.getvalue())
    if not txt:
        st.stop()

    meta, outs = run_pipeline(txt)

    # ---------------- display ----------------
    if meta:
        st.subheader("Study information (parsed)")
        st.json(meta)

    st.subheader("Extracted outcomes – hierarchical view")
    if outs:
        df = pd.DataFrame(outs)
        for col in ["outcome_domain", "outcome_specific", "definition", "timepoint"]:
            if col not in df.columns:
                df[col] = ""
        for dom in df["outcome_domain"].unique():
            st.markdown(f"**DOMAIN:** {dom}")
            for spec in df[(df["outcome_domain"] == dom) & (df["outcome_specific"] != dom)]["outcome_specific"].unique():
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {spec}")
            st.write("")

        # ---------- CSV export with metadata rows ----------
        st.subheader("Publication‑ready CSV export (with study metadata)")

        def meta_row(label: str, value: str) -> dict:
            return {"Domain": label, "Specific Outcome": value or "", "Definition": "", "Timepoint": ""}

        # Build metadata rows in the required order
        meta_rows: List[dict] = []
        if meta:
            m = meta
            meta_rows.extend([
                meta_row("Last author + year", f"{m.get('last_author_surname', '')} {m.get('publication_year', '')}"),
                meta_row("Paper title", m.get("paper_title", "")),
                meta_row("Journal", m.get("journal", "")),
                meta_row("Health‑care setting", m.get("health_care_setting", "")),
                meta_row("Country recruited", m.get("study_country", "")),
                meta_row("Patient population", m.get("patient_population", "")),
                meta_row("Targeted condition (definition)", f"{m.get('targeted_condition', '')} ({m.get('diagnostic_criteria', '')})".strip()),
                meta_row("Intervention tested", m.get("interventions_tested", "")),
                meta_row("Comparator", m.get("comparison_group", "")),
            ])

        # Outcome rows (same as before)
        outcome_rows: List[dict] = []
        for dom in df["outcome_domain"].unique():
            dom_row = df[df["outcome_domain"] == dom].iloc[0]
            outcome_rows.append({"Domain": dom, "Specific Outcome": "", "Definition": dom_row.get("definition", ""), "Timepoint": dom_row.get("timepoint", "")})
            for spec in df[(df["outcome_domain"] == dom) & (df["outcome_specific"] != dom)]["outcome_specific"].unique():
                outcome_rows.append({"Domain": "", "Specific Outcome": spec, "Definition": "", "Time
