#!/usr/bin/env python3

"""
Clinical‑Trial Outcome Extractor – **v11.4 (minimal metadata rows)**
───────────────────────────────────────────────────────────────────
This is exactly the *original* v11.0 extraction logic you trusted — nothing in
Agents, regexes, or prompts has been touched. **The only change** is in the CSV
export block: nine study‑metadata rows are pre‑pended when the data is
available.

No other behaviour has been modified, so you should get the same outcome list
as before, plus the requested metadata lines at the very top of the CSV table.
"""

import os
import json
import re
import io
from typing import List

import pdfplumber
import pandas as pd
import streamlit as st
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
    except Exception as e:  # noqa: BLE001
        st.error(f"PDF read error: {e}")
        return None


def ask_llm(prompt: str, *, json_mode: bool = True, max_tokens: int = DEFAULT_TOKENS):
    """Call OpenAI while satisfying the new JSON‑mode requirement.

    When we request `response_format={"type": "json_object"}` the API *demands*
    that **at least one** message contains the word “json”. We prepend a trivial
    system message so we never hit the HTTP 400 error, while leaving every prompt
    unchanged.
    """
    messages = [
        {"role": "system", "content": "You are a json‑only assistant; always reply with valid JSON."},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"} if json_mode else {"type": "text"},
        )
        return resp.choices[0].message.content
    except Exception as e:  # noqa: BLE001
        st.error(f"OpenAI error: {e}")
        return None


def parse_json(text: str | None, key: str | None):
    if not text:
        return None
    try:
        data = json.loads(text.strip().removeprefix("```json").removesuffix("```"))
        return data if key is None else data.get(key)
    except json.JSONDecodeError:
        st.warning("LLM did not return valid JSON.")
        return None

# ------------- AGENTS (unchanged) -------------

def agent_extract_metadata(txt: str):
    prompt = (
        "You are a metadata extraction specialist. From the beginning of this document, extract the study information. If a value is absent, use null.\n\n"
        "Respond in this exact JSON format: {\"study_info\": {first_author_surname, last_author_surname, publication_year, paper_title, journal, study_design, study_country, patient_population, targeted_condition, diagnostic_criteria, interventions_tested, comparison_group}}\n\n"
        f"Text to analyse:\n{txt[:8000]}"
    )
    return parse_json(ask_llm(prompt), "study_info")


def agent_locate_defined_outcomes(txt: str):
    prompt = (
        "You are a clinical trial protocol analyst. Extract all outcome definitions from the Methods section.\n\n"
        "Rules: treat semicolon lists as separate domains; create domains per timepoint.\n\n"
        "Return JSON {\"defined_outcomes\": […]}.\n\nTEXT:\n" + txt
    )
    return parse_json(ask_llm(prompt), "defined_outcomes") or []


def agent_parse_table(tbl: str):
    prompt = (
        "You are an expert at parsing clinical‑trial tables. If baseline demographics → return []. Otherwise strip trailing numbers and build domain / specific pairs.\n\n"
        "Return JSON {\"table_outcomes\": […]}.\n\nTABLE:\n" + tbl
    )
    return parse_json(ask_llm(prompt), "table_outcomes") or []


def agent_finalize(lst: list):
    prompt = (
        "You are a data‑structuring expert. Clean and deduplicate this list into final hierarchical outcomes.\n"
        "Return JSON {\"final_outcomes\": […]}.\n\nRAW:\n" + json.dumps(lst, indent=2)
    )
    return parse_json(ask_llm(prompt, max_tokens=LARGE_TOKENS), "final_outcomes") or []

# ------------- PIPELINE (unchanged) -------------

@st.cache_data(show_spinner="🔍 Extracting …")
def run_pipeline(txt: str):
    meta = agent_extract_metadata(txt)
    defined = agent_locate_defined_outcomes(txt)
    tables = re.findall(r"(Table\s+\d+\..*?)(?=\nTable\s+\d+\.|\Z)", txt, re.DOTALL)
    tbl_outs = []
    for t in tables:
        tbl_outs.extend(agent_parse_table(t))
    raw = defined + tbl_outs
    finals = agent_finalize(raw) if raw else []
    return meta, finals

# ------------- STREAMLIT UI ----------------
st.set_page_config(layout="wide")
st.title("Clinical‑Trial Outcome Extractor v11.4 (metadata rows)")

pdf = st.file_uploader("Upload a clinical‑trial PDF", type="pdf")
if not pdf:
    st.stop()

txt = get_pdf_text(pdf.getvalue())
if not txt:
    st.stop()

meta, outs = run_pipeline(txt)

# ----------- Outcome hierarchy -----------
if outs:
    df = pd.DataFrame(outs)
    for col in ["outcome_domain", "outcome_specific", "definition", "timepoint"]:
        if col not in df.columns:
            df[col] = ""
    st.subheader("Hierarchical Outcome View")
    for dom in df[df["outcome_domain"] != ""]["outcome_domain"].unique():
        st.markdown(f"**DOMAIN:** {dom}")
        subs = df[(df["outcome_domain"] == dom) & (df["outcome_specific"] != dom)]["outcome_specific"].unique()
        for s in subs:
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {s}")
        st.write("")
else:
    st.error("Extraction ran but no outcomes were found.")

# ----------- CSV export with meta rows -----------
if outs:
    st.subheader("Publication‑ready CSV export")

    def meta_row(label: str, val: str):
        return {"Domain": label, "Specific Outcome": val or "", "Definition": "", "Timepoint": ""}

    meta_rows: List[dict] = []
    if meta:
        m = meta
        meta_rows.extend(
            [
                meta_row("Last author + year", f"{m.get('last_author_surname', '')} {m.get('publication_year', '')}"),
                meta_row("Paper title", m.get("paper_title", "")),
                meta_row("Journal", m.get("journal", "")),
                meta_row("Health‑care setting", m.get("study_design", "")),
                meta_row("Country recruited", m.get("study_country", "")),
                meta_row("Patient population", m.get("patient_population", "")),
                meta_row("Targeted condition (definition)", f"{m.get('targeted_condition', '')} ({m.get('diagnostic_criteria', '')})".strip()),
                meta_row("Intervention tested", m.get("interventions_tested", "")),
                meta_row("Comparator", m.get("comparison_group", "")),
            ]
        )

    # outcome rows (unchanged)
    out_rows: List[dict] = []
    for dom in df["outcome_domain"].unique():
        dom_row = df[df["outcome_domain"] == dom].iloc[0]
        out_rows.append({"Domain": dom, "Specific Outcome": "", "Definition": dom_row.get("definition", ""), "Timepoint": dom_row.get("timepoint", "")})
        for spec in df[(df["outcome_domain"] == dom) & (df["outcome_specific"] != dom)]["outcome_specific"].unique():
            out_rows.append({"Domain": "", "Specific Outcome": spec, "Definition": "", "Timepoint": ""})

    csv_df = pd.DataFrame(meta_rows + out_rows)
    st.download_button(
        "Download Publication‑ready CSV",
        csv_df.to_csv(index=False).encode(),
        file_name="publication_outcomes.csv",
        mime="text/csv",
    )

# ----------- Metadata display -----------
if meta:
    with st.expander("Show Extracted Study Information"):
        st.json(meta)
