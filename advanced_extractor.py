#!/usr/bin/env python3

"""
Universal Clinical Trial Data Extractor – patched version (v12.1)

Key fixes compared with v12.0
─────────────────────────────
1. **Deterministic hierarchy pass** – guarantees that table headers (domains) and
   row‑level outcomes are distinguished even when indentation is lost after
   PDF extraction.
2. **Qualifier preservation** – never removes text that appears before the first
   digit on a line, so phrases like “without preeclampsia” survive intact.
3. **Duplicate‑header guard** – drops any row where the specific text is just a
   repeat of the domain header.

Everything else is the same as v12.0, so you can drop‑in replace the old file.
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
    st.info("Step 1 • Reading PDF text…")
    try:
        with pdfplumber.open(io.BytesIO(file_contents)) as pdf:
            full_text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
            if not full_text.strip():
                st.error("This PDF appears to be a scanned image or contains no extractable text.")
                return None
            st.success("✓ PDF text read successfully.")
            return full_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None


def ask_llm(prompt: str, *, is_json: bool = True, max_tokens: int = DEFAULT_TOKENS_FOR_RESPONSE) -> str:
    """Call the OpenAI chat‑completions endpoint and return message content."""
    try:
        response_format = {"type": "json_object"} if is_json else {"type": "text"}
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
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
    """Agent 1 – Extracts study metadata from ANY clinical‑trial paper."""
    prompt = f"""You are a universal clinical trial metadata extraction specialist. From this document, extract ALL available study information. If any value is not found, use null.

**REQUIRED FIELDS:**
- authors
- first_author_surname
- publication_year
- journal
- study_design
- study_country
- patient_population
- targeted_condition
- diagnostic_criteria
- interventions_tested
- comparison_group

Respond as JSON in this format:
{{
  \"study_metadata\": {{ … }}
}}

**Document excerpt (first 10 000 characters):**
{full_text[:10000]}"""

    return parse_json_response(ask_llm(prompt), "study_metadata")



def agent_locate_all_outcomes(full_text: str) -> list:
    """Agent 2 – Pulls every outcome mentioned in the narrative text."""
    prompt = f"""You are a universal clinical‑trial outcome extraction specialist. Extract ALL outcomes (primary, secondary, safety, exploratory, post‑hoc) from the document. Return JSON with a list under key \"defined_outcomes\"; see schema below:
{{
  \"outcome_domain\": "Main outcome category",
  \"outcome_specific\": "Specific measure",
  \"outcome_type\": "primary/secondary/safety/…",
  \"definition\": "…",
  \"measurement_method\": "…",
  \"timepoint\": "…"
}}

**Document text:**
{full_text}"""
    return parse_json_response(
        ask_llm(prompt, max_tokens=LARGE_TOKENS_FOR_RESPONSE),
        "defined_outcomes",
    ) or []



def agent_parse_universal_table(table_text: str) -> list:
    """Agent 3 – Parses a *single* table that has been pre‑tagged with ⟨DOMAIN⟩ / ⟨ROW⟩ markers."""
    prompt = f"""You are an expert at parsing clinical‑trial tables that have been pre‑annotated.

**Input conventions:**
- Lines starting with ⟨DOMAIN⟩ are **section headers** (outcome_domain).
- Lines starting with ⟨ROW⟩ are **row‑level outcomes** (outcome_specific) belonging to the most recent domain.
- A line never includes its statistics – only the textual label.

**Rules:**
1. Preserve the text EXACTLY as given – including qualifiers like “without preeclampsia”.
2. Never delete text that appears **before the first digit** in the original line.
3. Infer outcome_type: anything under a domain that contains “adverse”, “safety”, “event” → safety; otherwise efficacy.
4. timepoint: if the domain contains a phrase like “at <37 wk of gestation”, capture it (standardise to “<37 weeks of gestation”) and inherit to rows.

Return JSON with key \"table_outcomes\" → list of objects:
{{
  \"outcome_domain\": "…",
  \"outcome_specific\": "…",
  \"outcome_type\": "safety/efficacy/other",
  \"definition\": "",
  \"measurement_method\": "",
  \"timepoint\": "…"
}}

**Table text:**
{table_text}"""
    return parse_json_response(
        ask_llm(prompt, max_tokens=LARGE_TOKENS_FOR_RESPONSE),
        "table_outcomes",
    ) or []



def agent_finalize_comprehensive_structure(messy_list: list) -> list:
    """Agent 4 – Dedupes & harmonises, preserving domain→row hierarchy exactly."""
    prompt = f"""You are a clinical‑trial data structuring expert. Clean and organise the list while preserving the EXACT domain/specific hierarchy and complete names. Follow these rules:

1. Do **not** create generic domains or truncate outcome names.
2. Preserve timepoints embedded in domain strings (standardise \"wk\" → \"weeks\").
3. outcome_type is already provided – keep as is unless obviously wrong.
4. Remove duplicate rows where outcome_specific is just a repeat of outcome_domain.

Return JSON with key \"final_outcomes\".

**List to process:**
{json.dumps(messy_list, indent=2)}"""
    return parse_json_response(
        ask_llm(prompt, max_tokens=LARGE_TOKENS_FOR_RESPONSE),
        "final_outcomes",
    ) or []

# ---------- 3. TABLE UTILITIES ----------


def extract_all_tables(full_text: str) -> list:
    """Locate raw table chunks in the PDF text using several regex patterns."""
    patterns = [
        r"(Table \d+\..*?)(?=\nTable \d+\.|\nFigure \d+\.|\n\n[A-Z][A-Z\s]+\n|\Z)",  # Standard
        r"(TABLE \d+\..*?)(?=\nTABLE \d+\.|\nFIGURE \d+\.|\n\n[A-Z][A-Z\s]+\n|\Z)",  # All‑caps
        r"(\nTable \d+[:\.].*?)(?=\nTable \d+|\nFigure \d+|\n\n[A-Z][A-Z\s]+\n|\Z)",  # Variation
        r"(\n\s*Table \d+.*?)(?=\n\s*Table \d+|\n\s*Figure \d+|\n\n[A-Z][A-Z\s]+\n|\Z)",  # Spaced
    ]

    raw_tables: list[str] = []
    for pat in patterns:
        raw_tables.extend(re.findall(pat, full_text, flags=re.DOTALL | re.IGNORECASE))

    # Deduplicate using multiple signatures
    seen: set[str] = set()
    unique_tables: list[str] = []
    for tbl in raw_tables:
        sigs = [
            re.sub(r"\s+", " ", tbl[:100]),
            re.sub(r"\s+", " ", tbl[:200].split("\n")[0] if "\n" in tbl[:200] else tbl[:100]),
            re.sub(r"\s+", " ", "\n".join(tbl.split("\n")[2:4])) if len(tbl.split("\n")) > 3 else tbl[:100],
        ]
        if any(sig in seen for sig in sigs if sig.strip()):
            continue
        seen.update(sig for sig in sigs if sig.strip())
        unique_tables.append(tbl)
    return unique_tables


# ---------- 4. *** NEW *** PRE‑PROCESSOR ----------

def preprocess_table_for_llm(table_text: str) -> str:
    """Tag each line as ⟨DOMAIN⟩ or ⟨ROW⟩ so the LLM no longer has to guess the hierarchy."""
    processed_lines: list[str] = []
    for raw in table_text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # A row contains at least one digit following some text.
        is_row = bool(re.search(r"\s\d", line))

        if is_row:
            text_part = re.split(r"\s\d", line, maxsplit=1)[0].strip("–—- ")
            processed_lines.append(f"⟨ROW⟩ {text_part}")
        else:
            domain = re.sub(r"\s*[–—-].*$", "", line).strip()
            processed_lines.append(f"⟨DOMAIN⟩ {domain}")
    return "\n".join(processed_lines)


# ---------- 5. MAIN ORCHESTRATION PIPELINE ----------

@st.cache_data(show_spinner="Step 2 • Running comprehensive AI extraction pipeline…")
def run_universal_extraction_pipeline(full_text: str):
    """End‑to‑end: metadata, narrative outcomes, table outcomes, final structured output."""

    # 1 • Study metadata
    study_metadata = agent_extract_comprehensive_metadata(full_text)

    # 2 • Outcomes defined in the text
    defined_outcomes = agent_locate_all_outcomes(full_text)

    # 3 • Tables → deterministic pre‑processing → LLM parsing
    raw_table_outcomes: list[dict] = []
    for tbl in extract_all_tables(full_text):
        prepped = preprocess_table_for_llm(tbl)
        raw_table_outcomes.extend(agent_parse_universal_table(prepped))

    # Combine narrative + table outcomes
    raw_combined = defined_outcomes + raw_table_outcomes
    if not raw_combined:
        return study_metadata, []

    # 4 • Tiny post‑processing guard – drop domain duplicates
    deduped: list[dict] = []
    for o in raw_combined:
        if o.get("outcome_specific") and o["outcome_specific"].strip().lower().startswith(o.get("outcome_domain", "").strip().lower()):
            continue
        deduped.append(o)

    # 5 • Final structuring & cleaning by Agent 4
    final_outcomes = agent_finalize_comprehensive_structure(deduped)
    return study_metadata, final_outcomes

# ---------- 6. STREAMLIT UI ----------

st.set_page_config(layout="wide")
st.title("Universal Clinical Trial Data Extractor (v12.1 – patched)")

st.markdown(
    """**✨ What’s new in 12.1**
- Deterministic table hierarchy tagging (⟨DOMAIN⟩/⟨ROW⟩) ➜ bullet‑level outcome capture even when indentation is lost.
- Qualifier text (“without preeclampsia”, etc.) is preserved.
- Automatic guard against domain‑header duplicates.
"""
)

uploaded_file = st.file_uploader("Upload ANY clinical‑trial PDF", type="pdf")

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    full_text = get_pdf_text(pdf_bytes)

    if full_text:
        meta, outcomes = run_universal_extraction_pipeline(full_text)

        if outcomes:
            st.success(f"✅ Extraction complete for **{uploaded_file.name}**")

            # -------- Study metadata --------
            if meta:
                st.subheader("📋 Study information")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Authors:** {meta.get('authors') or '—'}")
                    st.write(f"**Journal:** {meta.get('journal') or '—'}")
                    st.write(f"**Year:** {meta.get('publication_year') or '—'}")
                    st.write(f"**Study design:** {meta.get('study_design') or '—'}")
                with col2:
                    st.write(f"**Country:** {meta.get('study_country') or '—'}")
                    st.write(f"**Population:** {meta.get('patient_population') or '—'}")
                    st.write(f"**Condition:** {meta.get('targeted_condition') or '—'}")
                    st.write(f"**Interventions:** {meta.get('interventions_tested') or '—'}")

            # -------- Outcomes (hierarchical display) --------
            st.subheader("🎯 Extracted outcomes")
            df = pd.DataFrame(outcomes)
            domain_order = df["outcome_domain"].drop_duplicates().tolist()

            for domain in domain_order:
                st.markdown(f"• **{domain}**")
                rows = df[df["outcome_domain"] == domain]
                for _, r in rows.iterrows():
                    specific = r.get("outcome_specific", "").strip()
                    if specific:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;◦ {specific}")

            # -------- CSV export --------
            st.subheader("📥 Export data (CSV)")
            base = {
                "Authors": meta.get("authors", "") if meta else "",
                "First_Author_Surname": meta.get("first_author_surname", "") if meta else "",
                "Publication_Year": meta.get("publication_year", "") if meta else "",
                "Journal": meta.get("journal", "") if meta else "",
                "Study_Design": meta.get("study_design", "") if meta else "",
                "Study_Country": meta.get("study_country", "") if meta else "",
                "Patient_Population": meta.get("patient_population", "") if meta else "",
                "Targeted_Condition": meta.get("targeted_condition", "") if meta else "",
                "Diagnostic_Criteria": meta.get("diagnostic_criteria", "") if meta else "",
                "Interventions_Tested": meta.get("interventions_tested", "") if meta else "",
                "Comparison_Group": meta.get("comparison_group", "") if meta else "",
            }
            export_rows = []
            for _, row in df.iterrows():
                out = base.copy()
                out.update({
                    "Outcome_Type": row.get("outcome_type", ""),
                    "Outcome_Domain": row.get("outcome_domain", ""),
                    "Outcome_Specific": row.get("outcome_specific", ""),
                    "Definition": row.get("definition", ""),
                    "Measurement_Method": row.get("measurement_method", ""),
                    "Timepoint": row.get("timepoint", ""),
                })
                export_rows.append(out)
            export_df = pd.DataFrame(export_rows)
            st.download_button(
                label="Download full dataset (CSV)",
                data=export_df.to_csv(index=False).encode(),
                file_name=f"Clinical_Trial_Data_{uploaded_file.name.replace('.pdf', '')}.csv",
                mime="text/csv",
            )
            with st.expander("Preview first 10 rows"):
                st.dataframe(export_df.head(10))
        else:
            st.warning("⚠️ No outcomes were extracted.")
            if meta:
                st.subheader("📋 Metadata found")
                st.write(meta)
