#!/usr/bin/env python3
# -------- advanced_extractor.py (v11.1 – Final, High-Quality) --------
"""
Clinical-Trial Outcome Extractor
Patch notes v11.1
- Integrated the most detailed, example-driven prompts for maximum extraction quality.
- Re-architected the export function to produce a single, clean, "publication-ready" CSV.
- This version is designed to be the final, definitive script.
"""

import os
import json
import re
import io
import pandas as pd
import streamlit as st
from openai import OpenAI
import pdfplumber

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

def agent_extract_metadata_and_definitions(full_text: str) -> tuple[dict | None, list]:
    """Agent 1 – Extracts metadata AND locates outcome definitions from the text."""
    prompt = f"""You are an expert medical reviewer. Analyze the full text of the clinical trial paper.
Extract two types of information:
1.  **study_info:** The overall study metadata (author, year, design, population, condition, interventions, comparison).
2.  **defined_outcomes:** All outcome definitions found in the text, especially in the 'Methods' section. Pay attention to definitions, measurement instruments, and timepoints. Handle semicolon-separated lists and time-based groupings as separate domains.

You must respond with a single JSON object containing both keys: "study_info" and "defined_outcomes".

DOCUMENT TEXT TO ANALYZE:
{full_text}"""
    
    response = parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), None)
    if response:
        return response.get("study_info"), response.get("defined_outcomes", [])
    return None, []


def agent_parse_table(table_text: str) -> list:
    """Agent 2 – parse a single table and extract outcome names with high fidelity."""
    prompt = f"""You are an expert at parsing clinical trial tables. Analyze the single table text below with high precision.

STEP 1: CLASSIFY THE TABLE
First, determine if this table describes **baseline patient characteristics** (e.g., 'Characteristics of the Participants', age, race) or **clinical trial outcomes** (results, events, complications, efficacy, safety).
- If it is a BASELINE table, you MUST return an empty list: `{{"table_outcomes": []}}`

STEP 2: EXTRACT OUTCOMES (only if it is an outcome table)
- **Hierarchy:** A bolded heading or a line item that has other items indented under it is an 'outcome_domain'. The items listed under that heading are its 'specific_outcomes'.
- **Clean Names:** The outcome name is the text description ONLY. You MUST strip away all trailing data, numbers, percentages, and formatting like '— no. (%)' or 'no./total no. (%)'.

DETAILED HIERARCHY EXAMPLE:
- INPUT TEXT:
'''
Adverse outcomes at <34 wk of gestation
Any — no. (%) 32 (4.0)
Preeclampsia — no. (%) 3 (0.4)
Small-for-gestational-age status without preeclampsia — no./total no. (%) 7/785 (0.9)
'''
- REQUIRED JSON OUTPUT:
`{{
  "table_outcomes": [
    {{"outcome_domain": "Adverse outcomes at <34 wk of gestation", "outcome_specific": "Any"}},
    {{"outcome_domain": "Adverse outcomes at <34 wk of gestation", "outcome_specific": "Preeclampsia"}},
    {{"outcome_domain": "Adverse outcomes at <34 wk of gestation", "outcome_specific": "Small-for-gestational-age status without preeclampsia"}}
  ]
}}`

Respond with a JSON object with a list called "table_outcomes".

TABLE TEXT TO PARSE:
{table_text}"""
    return parse_json_response(ask_llm(prompt), "table_outcomes") or []

def agent_finalize_and_structure(messy_list: list) -> list:
    """Agent 3 – clean, dedupe and structure the combined outcome list."""
    prompt = f"""You are a data structuring expert. Clean, deduplicate, and structure this messy list of outcomes into a final hierarchical list.

RULES:
1. For each unique outcome domain, create one entry with `"outcome_type": "domain"`.
2. For each specific outcome under that domain, create a separate entry with `"outcome_type": "specific"`.
3. Combine information. If you see the same outcome multiple times, merge any definitions or timepoints.
4. Remove any obvious non-outcome entries or clear duplicates.

The final output must be a JSON object with a key 'final_outcomes'. Each item in the list must have the keys: 'outcome_type', 'outcome_domain', 'outcome_specific', 'definition', and 'timepoint'.

MESSY LIST TO PROCESS:
{json.dumps(messy_list, indent=2)}"""
    return parse_json_response(
        ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "final_outcomes"
    ) or []

# ---------- 3. MAIN ORCHESTRATION PIPELINE ----------

@st.cache_data(show_spinner="Step 2 / 3 – Running AI extraction pipeline…")
def run_extraction_pipeline(full_text: str):
    """Orchestrate the calls to the specialist agents."""
    
    study_info, defined_outcomes = agent_extract_metadata_and_definitions(full_text)
    
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

st.set_page_config(layout="wide", page_title="Clinical Trial Outcome Extractor")
st.title("Clinical-Trial Outcome Extractor")
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
                # Find the primary entry for the domain to get its definition/timepoint
                domain_df = df[(df["outcome_domain"] == domain) & (df["outcome_type"] == "domain")]
                if domain_df.empty: 
                    # If no specific domain entry, take the first row as representative
                    domain_row = df[df["outcome_domain"] == domain].iloc[0]
                else:
                    domain_row = domain_df.iloc[0]

                export_rows.append({"Domain": domain, "Specific Outcome": "", "Definition": domain_row.get("definition", ""),"Timepoint": domain_row.get("timepoint", "")})
                
                # Find all specific outcomes for this domain
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
        else:
            st.error("Extraction ran but *no* outcomes were found.")