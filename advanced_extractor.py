#!/usr/bin/env python3

"""
Universal Clinical Trial Data Extractor – patched version (v12.2)

Changelog vs 12.1
─────────────────
* **Fix 1 – OpenAI 400 error** · `ask_llm()` now always adds a *system* message
  that contains the lowercase word “json” whenever `response_format={'type':
  'json_object'}` is requested. This satisfies the new API requirement.
* **Fix 2 – Robust list‑typing** · After `agent_locate_all_outcomes()` we coerce
  the result into a list so `defined_outcomes + all_table_outcomes` can never
  raise `TypeError`.

Everything else (deterministic hierarchy tagging, qualifier preservation, etc.)
remains unchanged from 12.1.
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
    """Extract raw text from a PDF (bytes) via pdfplumber."""
    st.info("Step 1 • Reading PDF text…")
    try:
        with pdfplumber.open(io.BytesIO(file_contents)) as pdf:
            full_text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
            if not full_text.strip():
                st.error("This PDF appears to be a scanned image or contains no extractable text.")
                return None
            st.success("✓ PDF text read successfully.")
            return full_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None


def ask_llm(prompt: str, *, is_json: bool = True, max_tokens: int = DEFAULT_TOKENS_FOR_RESPONSE) -> str:
    """Call the OpenAI chat endpoint.

    When `is_json` is **True** we request `response_format={'type':'json_object'}`
    *and* prepend a system message that explicitly mentions the word “json”, as
    required by the OpenAI API (otherwise the request is rejected with HTTP 400).
    """
    try:
        response_format = {"type": "json_object"} if is_json else {"type": "text"}
        messages = [
            {
                "role": "system",
                "content": "You are a json‑only answering assistant. Always return valid JSON and nothing else.",
            },
            {"role": "user", "content": prompt},
        ]
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An API error occurred: {e}")
        return None


def parse_json_response(text: str, key: str | None = None):
    """Safely parse a JSON string that may be wrapped in ```json markdown."""
    if not text:
        return None
    try:
        json_str = text.strip().removeprefix("```json").removesuffix("```")
        data = json.loads(json_str)
        return data if key is None else data.get(key)
    except (json.JSONDecodeError, AttributeError):
        st.warning("Could not parse a valid JSON response from the AI.")
        return None

# ---------- 2. UNIVERSAL SPECIALISED AGENTS ----------


def agent_extract_comprehensive_metadata(full_text: str) -> dict:
    """Agent 1 – Extract study metadata."""
    prompt = f"""You are a universal clinical‑trial metadata extractor. Return the following fields as JSON:
authors, first_author_surname, publication_year, journal, study_design, study_country, patient_population,
 targeted_condition, diagnostic_criteria, interventions_tested, comparison_group.

**Document excerpt (first 10 000 characters):**
{full_text[:10000]}"""
    return parse_json_response(ask_llm(prompt), "study_metadata")


def agent_locate_all_outcomes(full_text: str) -> list:
    """Agent 2 – Extract all narrative‑text outcomes."""
    prompt = f"""You are an outcome‑extraction specialist. List every outcome (primary, secondary, safety, exploratory, post‑hoc) found in the paper.
Return a JSON object with a single key \"defined_outcomes\" whose value is a list of outcome objects.

**Document text:**
{full_text}"""
    return parse_json_response(
        ask_llm(prompt, max_tokens=LARGE_TOKENS_FOR_RESPONSE),
        "defined_outcomes",
    ) or []


def agent_parse_universal_table(table_text: str) -> list:
    """Agent 3 – Parse a pre‑tagged table (⟨DOMAIN⟩ / ⟨ROW⟩)."""
    prompt = f"""You receive a table where each line starts with either ⟨DOMAIN⟩ or ⟨ROW⟩.
Return JSON with key \"table_outcomes\" → list of objects preserving the text exactly.

**Table text:**
{table_text}"""
    return parse_json_response(
        ask_llm(prompt, max_tokens=LARGE_TOKENS_FOR_RESPONSE),
        "table_outcomes",
    ) or []


def agent_finalize_comprehensive_structure(messy_list: list) -> list:
    """Agent 4 – Deduplicate & final‑clean while preserving hierarchy."""
    prompt = f"""You are a clinical‑trial data structuring expert. Clean and deduplicate the list while *preserving* the exact domain/specific hierarchy.
Return JSON with key \"final_outcomes\".

**List to process:**
{json.dumps(messy_list, indent=2)}"""
    return parse_json_response(
        ask_llm(prompt, max_tokens=LARGE_TOKENS_FOR_RESPONSE),
        "final_outcomes",
    ) or []

# ---------- 3. TABLE UTILITIES ----------


def extract_all_tables(full_text: str) -> list:
    """Locate raw table chunks in the plain text using several regex patterns."""
    patterns = [
        r"(Table \d+\..*?)(?=\nTable \d+\.|\nFigure \d+\.|\n\n[A-Z][A-Z\s]+\n|\Z)",
        r"(TABLE \d+\..*?)(?=\nTABLE \d+\.|\nFIGURE \d+\.|\n\n[A-Z][A-Z\s]+\n|\Z)",
        r"(\nTable \d+[:\.].*?)(?=\nTable \d+|\nFigure \d+|\n\n[A-Z][A-Z\s]+\n|\Z)",
        r"(\n\s*Table \d+.*?)(?=\n\s*Table \d+|\n\s*Figure \d+|\n\n[A-Z][A-Z\s]+\n|\Z)",
    ]
    raw_tables: list[str] = []
    for pat in patterns:
        raw_tables.extend(re.findall(pat, full_text, flags=re.DOTALL | re.IGNORECASE))
    # Deduplicate
    seen: set[str] = set()
    uniq: list[str] = []
    for tbl in raw_tables:
        key = re.sub(r"\s+", " ", tbl[:200])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(tbl)
    return uniq

# ---------- 4. PRE‑PROCESSOR ----------

def preprocess_table_for_llm(table_text: str) -> str:
    """Tag each line so the LLM sees explicit hierarchy."""
    out: list[str] = []
    for raw in table_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        is_row = bool(re.search(r"\s\d", line))
        if is_row:
            text = re.split(r"\s\d", line, maxsplit=1)[0].strip("–—- ")
            out.append(f"⟨ROW⟩ {text}")
        else:
            domain = re.sub(r"\s*[–—-].*$", "", line).strip()
            out.append(f"⟨DOMAIN⟩ {domain}")
    return "\n".join(out)

# ---------- 5. MAIN PIPELINE ----------

@st.cache_data(show_spinner="Step 2 • Running extraction pipeline…")
def run_universal_extraction_pipeline(full_text: str):
    """Metadata → narrative outcomes → table outcomes → final structured list."""
    study_meta = agent_extract_comprehensive_metadata(full_text)

    defined_outcomes = agent_locate_all_outcomes(full_text) or []
    # Ensure list‑type
    if isinstance(defined_outcomes, dict):
        defined_outcomes = [defined_outcomes]

    table_outcomes: list[dict] = []
    for raw_tbl in extract_all_tables(full_text):
        prepped = preprocess_table_for_llm(raw_tbl)
        table_outcomes.extend(agent_parse_universal_table(prepped))

    raw_combined = defined_outcomes + table_outcomes
    if not raw_combined:
        return study_meta, []

    final_outcomes = agent_finalize_comprehensive_structure(raw_combined)
    return study_meta, final_outcomes

# ---------- 6. STREAMLIT UI ----------

st.set_page_config(layout="wide")
st.title("Universal Clinical Trial Data Extractor (v12.2)")

st.markdown(
    """**What’s new (12.2)**
* OpenAI 400‑error fixed (system msg with “json”).
* Safer handling when narrative extractor returns a dict instead of list.
"""
)

uploaded_file = st.file_uploader("Upload a clinical‑trial PDF", type="pdf")

if uploaded_file:
    full_text = get_pdf_text(uploaded_file.getvalue())
    if full_text:
        meta, outcomes = run_universal_extraction_pipeline(full_text)
        if outcomes:
            st.success(f"✅ Extraction complete for **{uploaded_file.name}**")
            # Display & export (same as before)…
            # ——— Study info ———
            if meta:
                st.subheader("Study information")
                for k, v in meta.items():
                    st.write(f"**{k.replace('_', ' ').title()}:** {v or '—'}")
            # ——— Outcomes ———
            st.subheader("Extracted outcomes")
            df = pd.DataFrame(outcomes)
            for domain, grp in df.groupby("outcome_domain"):
                st.markdown(f"• **{domain}**")
                for _, r in grp.iterrows():
                    if r["outcome_specific"]:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;◦ {r['outcome_specific']}")
            # ——— CSV export ———
            st.subheader("Export CSV")
            base = {k.title().replace('_', ' '): v for k, v in (meta or {}).items()}
            rows = []
            for _, r in df.iterrows():
                row = base.copy()
                row.update({
                    "Outcome Type": r.get("outcome_type", ""),
                    "Outcome Domain": r.get("outcome_domain", ""),
                    "Outcome Specific": r.get("outcome_specific", ""),
                    "Definition": r.get("definition", ""),
                    "Measurement Method": r.get("measurement_method", ""),
                    "Timepoint": r.get("timepoint", ""),
                })
                rows.append(row)
            csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode()
            st.download_button("Download CSV", csv_bytes, file_name=f"trial_{uploaded_file.name}.csv", mime="text/csv")
        else:
            st.warning("⚠️ No outcomes were extracted.")
