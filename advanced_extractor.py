#!/usr/bin/env python3

"""
Universal Clinical‑Trial Data Extractor – **v12.5**
──────────────────────────────────────────────────
Feature + polish release.

### New in 12.5
1. **Stat‑suffix scrubber** – automatically removes trailing statistical
   artefacts such as “— no. (%)”, “— no./total no. (%)”, or bare percentages from
   every `outcome_domain` and `outcome_specific` while preserving qualifiers
   (e.g. “without preeclampsia”).  You’ll never see those fragments in the UI or
   CSV again.
2. **CSV export is back** – one‑click download containing study metadata plus
   the cleaned outcome list. Preview of the first 10 rows is shown under an
   expander.
3. **Minor UI tidy‑up** – grouped outcomes now respect the cleaned labels.

All other logic (hierarchy tagging, validation, debug panel) is unchanged.
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
    try:
        with pdfplumber.open(io.BytesIO(file_contents)) as pdf:
            return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return None


def ask_llm(prompt: str, *, json_mode: bool = True, max_tokens: int = DEFAULT_TOKENS_FOR_RESPONSE):
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

# ---------- 2. STAT‑SUFFIX SCRUBBER -------------

def clean_label(label: str) -> str:
    """Drop trailing statistical artefacts while preserving qualifiers."""
    if not label:
        return label
    # Remove patterns like "— no. (%)" or "— no./total no. (%)"
    label = re.sub(r"\s*[–—-]\s*(no\.|events|cases).*", "", label, flags=re.I)
    # Remove trailing counts in parentheses e.g. "(1.7)" or "(14%)"
    label = re.sub(r"\s*\(\d+(?:\.\d+)?%?\)$", "", label)
    return label.strip()

# ---------- 3. AGENTS ----------

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

# ---------- 4. TABLE UTILITIES ----------

def find_tables(txt: str):
    pat = r"(Table \d+\..*?)(?=\nTable \d+\.|
                             \nFigure \d+\.|
                             \Z)"  # single‑line concise regex
    return re.findall(pat, txt, re.DOTALL | re.VERBOSE)


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

# ---------- 5. ORCHESTRATOR ----------

@st.cache_data(show_spinner="🔍 Extracting…")
def run_pipeline(txt: str):
    meta = agent_extract_metadata(txt)
    narrative = agent_locate_outcomes(txt)
    tables = []
    for raw in find_tables(txt):
        tables.extend(agent_parse_table(tag_lines(raw)))
    combined = narrative + tables
    finals = agent_finalize(combined) if combined else []

    # 🔒 HARD VALIDATION & CLEANING
    for obj in finals:
        domain = clean_label(obj.get("outcome_domain", ""))
        specific = clean_label(obj.get("outcome_specific", ""))
        if not domain and not specific:
            domain = specific = "Unlabelled"
        obj["outcome_domain"] = domain or "Unlabelled"
        obj["outcome_specific"] = specific or "Unlabelled"
    return meta, finals

# ---------- 6. STREAMLIT UI ----------

st.set_page_config(layout="wide")
st.title("Universal Clinical‑Trial Extractor v12.5")

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

        # — CSV export —
        st.subheader("CSV export")
        meta_base = {k.title().replace("_", " "): v for k, v in (meta or {}).items()}
        rows = []
        for _, r in df.iterrows():
            row = meta_base.copy()
            row.update({
                "Outcome Type": r.get("outcome_type", ""),
                "Outcome Domain": r.get("outcome_domain", ""),
                "Outcome Specific": r.get("outcome_specific", ""),
                "Definition": r.get("definition", ""),
                "Measurement Method": r.get("measurement_method", ""),
                "Timepoint": r.get("timepoint", ""),
            })
            rows.append(row)
        export_df = pd.DataFrame(rows)
        st.download_button(
            "Download cleaned outcomes as CSV",
            export_df.to_csv(index=False).encode(),
            file_name="clinical_trial_outcomes.csv",
            mime="text/csv",
        )
        with st.expander("Preview first 10 rows"):
            st.dataframe(export_df.head(10))
    else:
        st.info("No outcomes extracted – check the debug panel or PDF quality.")
