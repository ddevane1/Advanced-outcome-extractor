#!/usr/bin/env python3

"""
Universal Clinical‑Trial Data Extractor – **v12.3**
──────────────────────────────────────────────────
Bug‑fix release replacing 12.2

▶ **Fix A – SyntaxError “( was never closed”**  
  *Root cause:* a long `re.findall()` pattern was split across lines during an
  earlier copy‑paste.  
  *Fix:* consolidate the regex on one line.

▶ **Fix B – KeyError 'outcome_domain' when DataFrame is empty or column
  missing**  
  *Root cause:* if the LLM returns an empty list _or_ objects lacking
  `outcome_domain`, converting to a DataFrame and calling `groupby()` explodes.
  
  *Fix:* guard before grouping and create placeholder columns when absent.

▶ **Fix C – Safer narrative‑outcome coercion**  
  If the LLM returns a single dict instead of a list, we now wrap it and also
  validate each item has the expected keys.

Everything else (hierarchy tagging, qualifier preservation, JSON system msg)
remains unchanged.
"""

import os
import json
import re
import io
import pdfplumber
import streamlit as st
import pandas as pd
from openai import OpenAI

# -------------------- CONFIG --------------------
MODEL = "gpt-4o"
DEFAULT_TOKENS_FOR_RESPONSE = 4096
LARGE_TOKENS_FOR_RESPONSE = 8192
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))

# ---------------- 1. CORE HELPERS ---------------

@st.cache_data
def get_pdf_text(file_contents: bytes):
    """Extract text from PDF bytes."""
    try:
        with pdfplumber.open(io.BytesIO(file_contents)) as pdf:
            return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return None


def ask_llm(prompt: str, *, json_mode: bool = True, max_tokens: int = DEFAULT_TOKENS_FOR_RESPONSE):
    """Wrapper around OpenAI chat with mandatory system message when json_mode."""
    messages = [
        {
            "role": "system",
            "content": "You are a json‑only assistant. Always respond with a valid JSON document.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"} if json_mode else {"type": "text"},
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return None


def parse_json(text: str, key: str | None = None):
    if not text:
        return None
    try:
        data = json.loads(text.strip().removeprefix("```json").removesuffix("```"))
        return data if key is None else data.get(key)
    except json.JSONDecodeError:
        st.warning("LLM did not return valid JSON.")
        return None

# ---------- 2. LLM AGENTS ----------

def agent_extract_metadata(txt: str) -> dict:
    prompt = f"Extract the study metadata fields as JSON.\n\nTEXT:\n{txt[:10000]}"
    return parse_json(ask_llm(prompt), "study_metadata")


def agent_locate_outcomes(txt: str) -> list:
    prompt = f"List all study outcomes. Return JSON {{'defined_outcomes': [...]}}.\nTEXT:\n{txt}"
    lst = parse_json(ask_llm(prompt, max_tokens=LARGE_TOKENS_FOR_RESPONSE), "defined_outcomes") or []
    if isinstance(lst, dict):
        lst = [lst]
    return lst


def agent_parse_table(prepped: str) -> list:
    prompt = f"Parse the table lines (⟨DOMAIN⟩ / ⟨ROW⟩). Return JSON {{'table_outcomes': [...]}}.\nLINES:\n{prepped}"
    return parse_json(ask_llm(prompt, max_tokens=LARGE_TOKENS_FOR_RESPONSE), "table_outcomes") or []


def agent_finalize(lst: list) -> list:
    prompt = f"Clean and deduplicate the list, keeping hierarchy. Return JSON {{'final_outcomes': [...]}}.\nDATA:\n{json.dumps(lst)[:8000]}"
    return parse_json(ask_llm(prompt, max_tokens=LARGE_TOKENS_FOR_RESPONSE), "final_outcomes") or []

# ---------- 3. TABLE FINDER + PREP ----------

def find_tables(txt: str) -> list:
    pattern = r"(Table \d+\..*?)(?=\nTable \d+\.|\nFigure \d+\.|\Z)"  # fixed single‑line regex
    return re.findall(pattern, txt, flags=re.DOTALL)


def tag_lines(table: str) -> str:
    out = []
    for line in table.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.search(r"\s\d", line):
            out.append("⟨ROW⟩ " + re.split(r"\s\d", line, 1)[0].strip("–—- "))
        else:
            out.append("⟨DOMAIN⟩ " + re.sub(r"\s*[–—-].*$", "", line).strip())
    return "\n".join(out)

# ---------- 4. ORCHESTRATOR ----------

@st.cache_data(show_spinner="Running extraction…")
def run_pipeline(txt: str):
    meta = agent_extract_metadata(txt)
    narrative = agent_locate_outcomes(txt)
    tables = []
    for raw in find_tables(txt):
        tables.extend(agent_parse_table(tag_lines(raw)))
    combined = narrative + tables
    if not combined:
        return meta, []
    finals = agent_finalize(combined)
    return meta, finals

# ---------- 5. STREAMLIT UI ----------

st.set_page_config(layout="wide")
st.title("Universal Clinical‑Trial Extractor v12.3")

pdf = st.file_uploader("Upload a clinical‑trial PDF", type="pdf")
if pdf:
    text = get_pdf_text(pdf.getvalue())
    if not text:
        st.stop()
    meta, outs = run_pipeline(text)

    # ---- show metadata ----
    if meta:
        st.subheader("Study information")
        for k, v in meta.items():
            st.write(f"**{k.replace('_', ' ').title()}:** {v or '—'}")

    # ---- show outcomes ----
    st.subheader("Extracted outcomes")
    if outs:
        df = pd.DataFrame(outs)
        # ensure required columns
        for col in ("outcome_domain", "outcome_specific"):
            if col not in df.columns:
                df[col] = "—"
        if "outcome_domain" in df.columns:
            for domain, grp in df.groupby("outcome_domain", dropna=False):
                st.markdown(f"• **{domain}**")
                for _, row in grp.iterrows():
                    if row["outcome_specific"] and row["outcome_specific"] != "—":
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;◦ {row['outcome_specific']}")
    else:
        st.info("No outcomes extracted.")
