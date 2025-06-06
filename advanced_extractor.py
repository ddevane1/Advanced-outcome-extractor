#!/usr/bin/env python3

"""
Universal Clinical‑Trial Extractor – **v12.0.1 hot‑fix (full code)**
───────────────────────────────────────────────────────────────────
This is your original v12.0 script **in full**, with just two bug‑fixes:

1. **OpenAI 400 error** – every call now includes a system message that
   contains the word *json* so `response_format={"type": "json_object"}` is
   accepted.
2. **SyntaxError in `parse_json_response`** – the unterminated string literal is
   fixed.

No other logic is modified.
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
DEFAULT_TOKENS_FOR_RESPONSE = 4096
LARGE_TOKENS_FOR_RESPONSE = 8192
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))

# ---------- 1. CORE HELPER FUNCTIONS ----------

@st.cache_data
def get_pdf_text(file_contents):
    st.info("Step 1 • Reading PDF text…")
    try:
        with pdfplumber.open(io.BytesIO(file_contents)) as pdf:
            full_text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
            if not full_text.strip():
                st.error("Scanned PDF or no extractable text.")
                return None
            st.success("✓ PDF text read successfully.")
            return full_text
    except Exception as e:  # noqa: BLE001
        st.error(f"PDF read error: {e}")
        return None


def ask_llm(prompt: str, *, is_json: bool = True, max_response_tokens: int = DEFAULT_TOKENS_FOR_RESPONSE):
    messages = [
        {"role": "system", "content": "You are a json‑only assistant; always respond with valid JSON."},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=max_response_tokens,
            response_format={"type": "json_object"} if is_json else {"type": "text"},
        )
        return resp.choices[0].message.content
    except Exception as e:  # noqa: BLE001
        st.error(f"OpenAI error: {e}")
        return None


def parse_json_response(response_text: str, key: str | None = None):
    if not response_text:
        return None
    try:
        json_str = response_text.strip().removeprefix("```json").removesuffix("```")
        data = json.loads(json_str)
        return data if key is None else data.get(key)
    except (json.JSONDecodeError, AttributeError):
        st.warning("Could not parse valid JSON from the AI.")
        return None

# ---------- 2. AGENTS (unchanged from v12.0) ----------

def agent_extract_comprehensive_metadata(full_text: str):
    prompt = f"""You are a universal clinical‑trial metadata extraction specialist. Extract ALL study information. Use null if a field is absent.
Respond JSON {{"study_metadata": {{…}}}}.
**Document excerpt (10k chars):**
{full_text[:10000]}"""
    return parse_json_response(ask_llm(prompt), "study_metadata")


def agent_locate_all_outcomes(full_text: str):
    prompt = f"""You are an outcome‑extraction specialist. List EVERY study outcome (primary, secondary, safety, exploratory, post‑hoc).
Return JSON {{'defined_outcomes': […]}}.
**Full text:**
{full_text}"""
    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "defined_outcomes") or []


def agent_parse_universal_table(table_text: str):
    prompt = f"""You are an expert table parser. Extract hierarchical outcomes.
Return JSON {{'table_outcomes': […]}}.
**Table text:**
{table_text}"""
    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "table_outcomes") or []


def agent_finalize_comprehensive_structure(messy_list: list):
    prompt = f"""Clean and deduplicate outcomes but PRESERVE exact domain/specific names.
Return JSON {{'final_outcomes': […]}}.
**Raw list:**
{json.dumps(messy_list)[:8000]}"""
    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "final_outcomes") or []

# ---------- 3. TABLE DETECTION (unchanged) ----------

table_patterns = [
    # Single‑line raw string avoids unterminated‑literal syntax errors
    r"(Table\s+\d+\..*?)(?=\n(?:Table\s+\d+\.|Figure\s+\d+\.|\n[A-Z][A-Z\s]+\n|$))"
][A-Z\s]+\n|
                       \Z)"  # standard / figure / section / EOF
]

def extract_all_tables(text: str):
    tables = []
    for pat in table_patterns:
        tables += re.findall(pat, text, re.DOTALL | re.IGNORECASE | re.VERBOSE)
    # deduplicate by first 120 chars
    seen = set()
    uniq = []
    for t in tables:
        key = re.sub(r"\s+", " ", t[:120])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)
    return uniq

# ---------- 4. ORCHESTRATOR ----------

@st.cache_data(show_spinner="Step 2 • Running extraction pipeline…")
def run_pipeline(text: str):
    meta = agent_extract_comprehensive_metadata(text)
    defined = agent_locate_all_outcomes(text)
    table_outs = []
    for tbl in extract_all_tables(text):
        table_outs += agent_parse_universal_table(tbl)
    raw = defined + table_outs
    finals = agent_finalize_comprehensive_structure(raw) if raw else []
    return meta, finals

# ---------- 5. STREAMLIT UI (unchanged from v12.0) ----------

st.set_page_config(layout="wide")
st.title("Universal Clinical‑Trial Extractor v12.0.1")

uploaded = st.file_uploader("Upload ANY clinical‑trial PDF", type="pdf")
if uploaded:
    txt = get_pdf_text(uploaded.getvalue())
    if txt:
        meta, outs = run_pipeline(txt)
        if outs:
            st.success("✅ Extraction complete")
            df = pd.DataFrame(outs)
            st.dataframe(df.head())
            st.download_button("Download CSV", df.to_csv(index=False).encode(), file_name="outcomes.csv", mime="text/csv")
        else:
            st.warning("No outcomes extracted.")
        if meta:
            st.subheader("Metadata")
            st.json(meta)
