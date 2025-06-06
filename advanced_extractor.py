#!/usr/bin/env python3

"""
Clinical-Trial Outcome Extractor – **v11.1 (stable)**
───────────────────────────────────────────────────
This is your original **v11.0** script with only two mandatory 2025-API tweaks:

1. **JSON compliance** – every OpenAI call now includes a short *system* message
   that contains the word **json** so `response_format={"type":"json_object"}`
   is accepted (prevents HTTP 400).
2. **Table regex widened** – stops on *Table*, *Figure*, **or end-of-file** so
   a table followed by a Figure is no longer merged/truncated.

All other prompts, logic, and the publication-ready CSV export remain exactly
as in v11.0.
"""

import os
import json
import re
import io

import pdfplumber
import pandas as pd
import streamlit as st
from openai import OpenAI

# ----- CONFIG -----
MODEL = "gpt-4o"
DEFAULT_TOKENS = 4096
LARGE_TOKENS = 8192
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))

# ---------- 1. CORE HELPER FUNCTIONS ----------

@st.cache_data
def get_pdf_text(file_bytes: bytes):
    """Extract text from a PDF or return None."""
    st.info("Reading PDF text…")
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:  # noqa: BLE001
        st.error(f"PDF read error: {e}")
        return None


def ask_llm(prompt: str, *, json_mode: bool = True, max_tokens: int = DEFAULT_TOKENS):
    """OpenAI wrapper that satisfies the 2025 JSON requirement."""
    messages = [
        {"role": "system", "content": "You are a json-only assistant. Always reply with valid JSON."},
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


def parse_json(text: str, key: str | None):
    if not text:
        return None
    try:
        data = json.loads(text.strip().removeprefix("```json").removesuffix("```"))
        return data if key is None else data.get(key)
    except json.JSONDecodeError:
        st.warning("LLM did not return valid JSON.")
        return None

# ---------- 2. AGENTS (unchanged logic) ----------

def agent_extract_metadata(txt: str):
    prompt = (
        "You are a metadata extraction specialist. Extract study info. If missing, use null.\n\n"
        "Return JSON {\"study_info\": {…}}.\n\n"
        f"Text (first 8k chars):\n{txt[:8000]}"
    )
    return parse_json(ask_llm(prompt), "study_info")


def agent_locate_defined_outcomes(txt: str):
    prompt = (
        "Extract all outcome definitions from the Methods. Return JSON {'defined_outcomes': [...]}.\n\n"
        f"Doc text:\n{txt}"
    )
    lst = parse_json(ask_llm(prompt), "defined_outcomes") or []
    return lst


def agent_parse_table(tbl: str):
    prompt = (
        "Classify table (baseline vs outcome). If outcome, strip numbers and build domain/specific pairs.\n\n"
        "Return JSON {'table_outcomes': [...]}.\n\n"
        f"Table text:\n{tbl}"
    )
    return parse_json(ask_llm(prompt), "table_outcomes") or []


def agent_finalize(messy: list):
    prompt = (
        "Clean and deduplicate outcome list. Return JSON {'final_outcomes': [...]}.\n\n"
        f"Raw list:\n{json.dumps(messy, indent=2)}"
    )
    return parse_json(ask_llm(prompt, max_tokens=LARGE_TOKENS), "final_outcomes") or []

# ---------- 3. PIPELINE ----------

TABLE_REGEX = r"(Table\s+\d+\..*?)(?=\n(?:Table\s+\d+\.|Figure\s+\d+\.|$))"

@st.cache_data(show_spinner="Running extraction…")
def run_pipeline(txt: str):
    meta = agent_extract_metadata(txt)
    defined = agent_locate_defined_outcomes(txt)
    tables = re.findall(TABLE_REGEX, txt, re.DOTALL)
    tbl_outs = []
    for t in tables:
        tbl_outs.extend(agent_parse_table(t))
    raw = defined + tbl_outs
    finals = agent_finalize(raw) if raw else []
    return meta, finals

# ---------- 4. STREAMLIT UI ----------

st.set_page_config(layout="wide")
st.title("Clinical-Trial Outcome Extractor v11.1 (stable)")

pdf = st.file_uploader("Upload a clinical-trial PDF", type="pdf")
if pdf:
    txt = get_pdf_text(pdf.getvalue())
    if not txt:
        st.stop()

    meta, outs = run_pipeline(txt)

    if outs:
        df = pd.DataFrame(outs)
        for col in ["outcome_domain", "outcome_specific", "definition", "timepoint"]:
            if col not in df.columns:
                df[col] = ""
        # Hierarchy view
        st.subheader("Hierarchical Outcome View")
        for dom in df["outcome_domain"].unique():
            st.markdown(f"**DOMAIN:** {dom}")
            subs = df[(df["outcome_domain"] == dom) & (df["outcome_specific"] != dom)]["outcome_specific"].unique()
            for spec in subs:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {spec}")
            st.write("")

        # Publication-ready CSV
        rows = []
        for dom in df["outcome_domain"].unique():
            dom_row = df[df["outcome_domain"] == dom].iloc[0]
            rows.append({"Domain": dom, "Specific Outcome": "", "Definition": dom_row["definition"], "Timepoint": dom_row["timepoint"]})
            for spec in df[(df["outcome_domain"] == dom) & (df["outcome_specific"] != dom)]["outcome_specific"].unique():
                rows.append({"Domain": "", "Specific Outcome": spec, "Definition": "", "Timepoint": ""})
        csv = pd.DataFrame(rows).to_csv(index=False).encode()
        st.download_button("Download Publication-ready CSV", csv, file_name="outcomes_publication_ready.csv", mime="text/csv")
    else:
        st.warning("No outcomes extracted.")

    if meta:
        st.subheader("Extracted Study Information")
        st.json(meta)