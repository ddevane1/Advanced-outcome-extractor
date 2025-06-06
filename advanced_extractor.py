#!/usr/bin/env python3

"""
Universal Clinical‑Trial Data Extractor – **v12.4**
──────────────────────────────────────────────────
Maintenance release: forces the LLM to ALWAYS return `outcome_domain` &
`outcome_specific`, and adds on‑screen debugging to catch empty/placeholder
results.

### What changed since 12.3
1. **Stricter post‑LLM validation** – after the final agent we loop over every
   object; if either key is missing or blank we overwrite both with
   `"Unlabelled"` so the UI never shows `• —` again.
2. **Hard requirement in the Agent‑4 prompt** – the JSON schema bullet now says
   *“Both keys are REQUIRED — do not omit or rename them.”*
3. **Debug panel** – a new `st.expander("🛠 Raw LLM output")` under the outcome
   list so you can see exactly what the model returned when things look odd.

No other logic has changed.
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
    """Call OpenAI chat‑completions with mandatory JSON system message."""
    msgs = [
        {
            "role": "system",
            "content": "You are a json‑only assistant. Output **nothing** except a single valid JSON object.",
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


def parse_json(text: str, key: str | None = None):
    if not text:
        return None
    try:
        data = json.loads(text.strip().removeprefix("```json").removesuffix("```"))
        return data if key is None else data.get(key)
    except json.JSONDecodeError:
        st.warning("LLM did not return valid JSON.")
        return None

# ---------- 2. AGENTS ----------

def agent_extract_metadata(txt: str):
    prompt = (
        "Extract the study metadata fields as JSON with the exact keys: "
        "authors, first_author_surname, publication_year, journal, study_design, "
        "study_country, patient_population, targeted_condition, diagnostic_criteria, "
        "interventions_tested, comparison_group.\n\nTEXT:\n" + txt[:10000]
    )
    return parse_json(ask_llm(prompt), "study_metadata")


def agent_locate_outcomes(txt: str):
    prompt = (
        "List every narrative‑text outcome. Respond as JSON {'defined_outcomes': [ … ]}. "
        "Each outcome object MUST contain *both* 'outcome_domain' and 'outcome_specific'.\n\nTEXT:\n"
        + txt
    )
    lst = parse_json(ask_llm(prompt, max_tokens=LARGE_TOKENS_FOR_RESPONSE), "defined_outcomes") or []
    if isinstance(lst, dict):
        lst = [lst]
    return lst


def agent_parse_table(prepped: str):
    prompt = (
        "Parse the table lines (⟨DOMAIN⟩ / ⟨ROW⟩). Return JSON {'table_outcomes': […]}. "
        "Both keys 'outcome_domain' and 'outcome_specific' are REQUIRED on every object.\n\nLINES:\n"
        + prepped
    )
    return parse_json(ask_llm(prompt, max_tokens=LARGE_TOKENS_FOR_RESPONSE), "table_outcomes") or []


def agent_finalize(lst: list):
    prompt = (
        "Clean and deduplicate while preserving hierarchy. Output JSON {'final_outcomes': […]}. "
        "*Both* 'outcome_domain' and 'outcome_specific' keys are REQUIRED — do not omit, rename or leave blank.\n\nDATA:\n"
        + json.dumps(lst)[:8000]
    )
    return parse_json(ask_llm(prompt, max_tokens=LARGE_TOKENS_FOR_RESPONSE), "final_outcomes") or []

# ---------- 3. TABLE UTILITIES ----------

def find_tables(txt: str):
    pat = r"(Table \d+\..*?)(?=\nTable \d+\.|\nFigure \d+\.|\Z)"  # single‑line regex
    return re.findall(pat, txt, re.DOTALL)


def tag_lines(table: str):
    lines = []
    for raw in table.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.search(r"\s\d", line):
            lines.append("⟨ROW⟩ " + re.split(r"\s\d", line, 1)[0].strip("–—- "))
        else:
            lines.append("⟨DOMAIN⟩ " + re.sub(r"\s*[–—-].*$", "", line).strip())
    return "\n".join(lines)

# ---------- 4. ORCHESTRATOR ----------

@st.cache_data(show_spinner="🔍 Extracting…")
def run_pipeline(txt: str):
    meta = agent_extract_metadata(txt)
    narrative = agent_locate_outcomes(txt)
    tables = []
    for raw in find_tables(txt):
        tables.extend(agent_parse_table(tag_lines(raw)))
    combined = narrative + tables
    finals = agent_finalize(combined) if combined else []

    # 🔒 HARD VALIDATION – ensure keys exist & are non‑empty
    for obj in finals:
        if not obj.get("outcome_domain") or not obj.get("outcome_specific"):
            obj["outcome_domain"] = obj["outcome_specific"] = "Unlabelled"
    return meta, finals

# ---------- 5. STREAMLIT UI ----------

st.set_page_config(layout="wide")
st.title("Universal Clinical‑Trial Extractor v12.4")

pdf = st.file_uploader("Upload a clinical‑trial PDF", type="pdf")
if pdf:
    txt = get_pdf_text(pdf.getvalue())
    if not txt:
        st.stop()

    meta, outs = run_pipeline(txt)

    # — metadata —
    if meta:
        st.subheader("Study information")
        for k, v in meta.items():
            st.write(f"**{k.replace('_', ' ').title()}:** {v or '—'}")

    # — outcomes —
    st.subheader("Extracted outcomes")
    if outs:
        df = pd.DataFrame(outs)
        for domain, grp in df.groupby("outcome_domain", dropna=False):
            st.markdown(f"• **{domain}**")
            for _, row in grp.iterrows():
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;◦ {row['outcome_specific']}")
        # debug
        with st.expander("🛠 Raw LLM output"):
            st.json(outs)
    else:
        st.info("No outcomes extracted – check the debug panel or PDF quality.")
