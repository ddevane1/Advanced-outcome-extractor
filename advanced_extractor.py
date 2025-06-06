#!/usr/bin/env python3
# -------- advanced_extractor.py (v17.0 – Final, High-Quality) --------
"""
Clinical-Trial Outcome Extractor
Patch notes v17.0
- Restored detailed, example-driven prompts to all AI agents for high-quality extraction.
- Combined the robust stability of the v16 architecture with the detailed logic of earlier versions.
- This version is designed to be the final, production-ready script.
"""

import os
import json
import re
import io
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
def get_pdf_text(file_contents: bytes) -> str | None:
    """Extract text from PDF bytes and cache the result."""
    st.info("Step 1 / 3 – Reading PDF text…")
    try:
        with pdfplumber.open(io.BytesIO(file_contents)) as pdf:
            full_text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
        if not full_text.strip():
            st.error("This PDF appears to be scanned images only – no extractable text.")
            return None
        st.success("✓ PDF text read successfully")
        return full_text
    except Exception as exc:
        st.error(f"Error reading PDF: {exc}")
        return None

def ask_llm(prompt: str, max_response_tokens: int = DEFAULT_TOKENS_FOR_RESPONSE) -> str | None:
    """Generic function to call the OpenAI chat API in JSON-object mode."""
    final_prompt = "You must provide a response in a valid JSON object. " + prompt
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.0,
            max_tokens=max_response_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as exc:
        st.error(f"OpenAI API error: {exc}")
        return None

def parse_json_response(response_text: str | None, key: str | None):
    """Safely parse JSON and return either the whole dict or a key within it."""
    if not response_text: return None
    try:
        data = json.loads(response_text)
        return data if key is None else data.get(key)
    except (json.JSONDecodeError, AttributeError):
        st.warning("Could not parse valid JSON from AI response.")
        return None

# ---------- 2. SPECIALISED AGENT FUNCTIONS ----------

def agent_extract_metadata(full_text: str) -> dict | None:
    """Agent 1 – extract high-level study metadata."""
    prompt = f"""From the beginning of this document, extract the study information (author, year, journal, design, population, condition, intervention, comparison). If a value is absent, use null. Respond with a JSON object with a key "study_info".

Text to analyse:
{full_text[:8000]}"""
    return parse_json_response(ask_llm(prompt), "study_info")

def agent_locate_defined_outcomes(full_text: str) -> list:
    """Agent 2 – locate planned outcomes described in the Methods section."""
    prompt = f"""From the 'Methods' section of the document, extract all defined outcomes.
RULES:
1. Handle semicolon-separated lists as separate domains (e.g., 'A; B; and C' are three domains).
2. Handle time-based groupings as separate domains (e.g., '...before 34 weeks' and '...before 37 weeks').
Respond with a JSON object containing a list called "defined_outcomes".

Document text to analyse:
{full_text}"""
    return parse_json_response(ask_llm(prompt), "defined_outcomes") or []

def agent_parse_table(table_text: str) -> list:
    """Agent 3 – parse a single table and extract outcome names with high fidelity."""
    prompt = f"""You are an expert at parsing clinical trial tables. Analyze the single table text below.

STEP 1: CLASSIFY THE TABLE
First, determine if this table describes **baseline patient characteristics** (demographics, age) or **clinical trial outcomes** (results, events).
- If it is a BASELINE table, you MUST return an empty list: `{{"table_outcomes": []}}`

STEP 2: EXTRACT OUTCOMES (only if it is an outcome table)
- **Hierarchy:** A bolded heading is a 'domain'. Items listed under it are its 'specific outcomes'.
- **Clean Names:** The outcome name is the text description ONLY. You MUST strip away all trailing data, numbers, percentages, and formatting like '— no. (%)'.

DETAILED EXAMPLE:
- INPUT TEXT:
'''
Adverse outcomes at <34 wk of gestation
Any — no. (%) 32 (4.0)
Preeclampsia — no. (%) 3 (0.4)
'''
- REQUIRED JSON OUTPUT:
`{{
  "table_outcomes": [
    {{"outcome_domain": "Adverse outcomes at <34 wk of gestation", "outcome_specific": "Any"}},
    {{"outcome_domain": "Adverse outcomes at <34 wk of gestation", "outcome_specific": "Preeclampsia"}}
  ]
}}`

Respond with a JSON object with a list called "table_outcomes".

TABLE TEXT TO PARSE:
{table_text}"""
    return parse_json_response(ask_llm(prompt), "table_outcomes") or []

def agent_finalize_and_structure(messy_list: list) -> list:
    """Agent 4 – clean, dedupe and structure the combined outcome list."""
    prompt = f"""Clean, deduplicate, and structure this messy list of outcomes into a final hierarchical list. Create 'domain' and 'specific' outcome types. Also extract any definitions or timepoints mentioned. Respond with a JSON object with a key "final_outcomes".

MESSY LIST TO PROCESS:
{json.dumps(messy_list, indent=2)}"""
    return parse_json_response(
        ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "final_outcomes"
    ) or []

# ---------- 3. MAIN ORCHESTRATION PIPELINE ----------

@st.cache_data(show_spinner="Step 2 / 3 – running AI extraction pipeline…")
def run_extraction_pipeline(full_text: str):
    """Orchestrate the calls to the four specialist agents."""
    study_info = agent_extract_metadata(full_text)
    defined_outcomes = agent_locate_defined_outcomes(full_text)
    table_texts = re.findall(r"(Table \d+\..*?)(?=\nTable \d+\.|\Z)", full_text, re.DOTALL)
    all_table_outcomes: list = []
    if table_texts:
        for table_text in table_texts:
            parsed_outcomes = agent_parse_table(table_text)
            all_table_outcomes.extend(parsed_outcomes)
    
    raw_combined_list = defined_outcomes + all_table_outcomes
    if not raw_combined_list:
        return study_info, []

    final_outcomes = agent_finalize_and_structure(raw_combined_list)
    return study_info, final_outcomes

# ---------- 4. STREAMLIT UI ----------

st.set_page_config(layout="wide", page_title="Clinical Trial Outcome Extractor v17.0")
st.title("Clinical-Trial Outcome Extractor (v17.0)")
st.markdown("This tool uses a cached, multi-agent AI workflow to accurately and reliably extract outcomes from PDF clinical-trial reports.")

uploaded_file = st.file_uploader("Upload a PDF clinical-trial report to begin", type="pdf")

if uploaded_file is not None:
    file_contents = uploaded_file.getvalue()
    full_text = get_pdf_text(file_contents)

    if full_text:
        with st.spinner("Step 3 / 3 – Preparing final report…"):
            study_info, outcomes = run_extraction_pipeline(full_text)
        
        if outcomes:
            st.success(f"Processing complete for **{uploaded_file.name}**.")
            df = pd.DataFrame(outcomes)
            for col in ["outcome_domain", "outcome_specific", "outcome_type", "definition", "timepoint"]:
                if col not in df.columns: df[col] = ""
            df.fillna("", inplace=True)

            # HIERARCHICAL PRETTY PRINT
            st.subheader("Hierarchical Outcome View")
            domains = df[df["outcome_domain"].astype(str) != ""]["outcome_domain"].unique()
            for domain in domains:
                st.markdown(f"**DOMAIN:** {domain}")
                specific_outcomes = df[(df["outcome_domain"] == domain) & (df["outcome_specific"] != "") & (df["outcome_specific"] != domain)]["outcome_specific"].unique()
                if specific_outcomes.size:
                    for specific in specific_outcomes:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;• {specific}")
                else:
                    st.markdown("&nbsp;&nbsp;&nbsp;• *Primary outcome or domain with no specific sub-outcomes.*")
                st.write("")

            # EXPORT CSV
            st.subheader("Export Results")
            export_rows: list[dict[str, str]] = []
            for domain in domains:
                domain_df = df[df["outcome_domain"] == domain]
                if domain_df.empty: continue
                domain_row = domain_df.iloc[0]
                export_rows.append({"Domain": domain, "Specific Outcome": "", "Definition": domain_row.get("definition", ""),"Timepoint": domain_row.get("timepoint", "")})
                specific_df = df[(df["outcome_domain"] == domain) & (df["outcome_specific"] != "") & (df["outcome_specific"] != domain)]
                for _, row in specific_df.iterrows():
                    export_rows.append({"Domain": "", "Specific Outcome": row["outcome_specific"], "Definition": row.get("definition", ""),"Timepoint": row.get("timepoint", "")})
            export_df = pd.DataFrame(export_rows)

            st.download_button(
                label="**Download Publication-Ready CSV**",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"Publication_Outcomes_{uploaded_file.name}.csv",
                mime="text/csv"
            )

            # SIDE INFO
            with st.expander("Show Extracted Study Information"):
                st.json(study_info or {})
            with st.expander("Show Full Raw Data Table (for analysis)"):
                st.dataframe(df.astype(str))
        else:
            st.error("Extraction ran but *no* outcomes were found.")