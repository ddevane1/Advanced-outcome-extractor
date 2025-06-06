#!/usr/bin/env python3
# -------- advanced_extractor.py (v4.1 - final syntax verified) --------

import os
import json
import re
import streamlit as st
import pandas as pd
import pdfplumber
import tiktoken
from openai import OpenAI

# ----- CONFIG -----
MODEL = "gpt-4o"
TOKENS_FOR_RESPONSE = 4000
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))
enc = tiktoken.encoding_for_model(MODEL)

# ---------- 1. CORE HELPER FUNCTIONS ----------

def pdf_to_text(file):
    """Extracts text from an uploaded PDF file."""
    try:
        with pdfplumber.open(file) as pdf:
            full_text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
            if not full_text.strip():
                st.error("This PDF appears to be a scanned image or contains no extractable text.")
                return None
            return full_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def ask_llm(prompt: str, is_json: bool = True) -> str:
    """Generic function to call the OpenAI API."""
    try:
        response_format = {"type": "json_object"} if is_json else {"type": "text"}
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=TOKENS_FOR_RESPONSE,
            response_format=response_format
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An API error occurred: {e}")
        return None

def parse_json_response(response_text: str, key: str):
    """Safely parses JSON from the LLM response and extracts a key."""
    if not response_text:
        return None
    try:
        json_str = response_text.strip().removeprefix("```json").removesuffix("```")
        data = json.loads(json_str)
        return data if key is None else data.get(key)
    except (json.JSONDecodeError, AttributeError) as e:
        st.warning(f"Could not parse JSON for key '{key}'. Raw response: {response_text[:500]}")
        return None


# ---------- 2. SPECIALIZED AGENT FUNCTIONS ----------

def agent_extract_metadata(full_text: str) -> dict:
    """Agent 1: Extracts the high-level study metadata from the first few pages."""
    prompt = (
        "You are a metadata extraction specialist. From the beginning of this document, extract study information. If absent, use null.\n"
        'Respond in JSON: {"study_info": {"first_author_surname": "...", "publication_year": "...", "journal": "...", "study_design": "...", "study_country": "...", "patient_population": "...", "targeted_condition": "...", "diagnostic_criteria": "...", "interventions_tested": "...", "comparison_group": "..."}}\n\n'
        f"Text to analyze:\n{full_text[:6000]}"
    )
    st.write("↳ Agent 1: Extracting study metadata...")
    response = ask_llm(prompt)
    return parse_json_response(response, "study_info")

def agent_locate_defined_outcomes(full_text: str) -> list:
    """Agent 2: Finds planned outcomes from the Methods section using specific rules."""
    prompt = (
        "You are a clinical trial protocol analyst. Your task is to extract all outcome definitions, typically found in the 'Methods' section.\n\n"
        "**CRITICAL EXTRACTION RULES:**\n"
        "1.  **Handle Semicolon-Separated Lists:** When a sentence lists outcomes separated by semicolons (e.g., 'Secondary outcomes were A; B; and C'), treat each item (A, B, C) as a separate outcome domain.\n"
        "2.  **Handle Time-Based Grouping:** When outcomes are grouped by time (e.g., 'before 34 weeks'), create a separate domain for each timepoint.\n\n"
        "**EXAMPLE:**\n"
        "If you see: 'Secondary outcomes were adverse outcomes of pregnancy before 34 weeks of gestation; and poor fetal growth...'\n"
        "You MUST extract these as separate domains: 'Adverse outcomes of pregnancy before 34 weeks of gestation' and 'Poor fetal growth'.\n\n"
        "**OUTPUT FORMAT:** Return a JSON object with a list called 'defined_outcomes'.\n\n"
        f"**Document Text to Analyze:**\n{full_text}"
    )
    st.write("↳ Agent 2: Locating defined outcomes from text...")
    response = ask_llm(prompt)
    return parse_json_response(response, "defined_outcomes") or []


def agent_parse_table(table_text: str) -> list:
    """Agent 3: A specialist agent to parse the text of a single table."""
    prompt = (
        "You are an expert at parsing clinical trial tables. Your only job is to analyze the text of the single table provided below and extract its hierarchical outcomes. DO NOT HALLUCINATE.\n\n"
        "**CRITICAL TABLE PARSING RULES:**\n"
        "1.  **Identify the Domain:** The main heading of a group of outcomes is the 'outcome_domain'. It often ends with '— no. (%)' or is a bolded header for a section.\n"
        "2.  **Identify Specific Outcomes:** Items listed or indented under a domain heading are the 'outcome_specific' measures.\n"
        "3.  **Verbatim Extraction:** Use the EXACT wording from the table.\n\n"
        "--- TABLE PARSING EXAMPLES ---\n"
        "**EXAMPLE 1:**\n"
        "If the table text is:\n"
        "'''\nDeath or complications — no. (%)\n"
        "  Any                                32 (4.0)\n"
        "  Miscarriage, stillbirth, or death  19 (2.4)\n'''\n"
        "Extract as:\n"
        "- Domain: 'Death or complications', Specific: 'Any'\n"
        "- Domain: 'Death or complications', Specific: 'Miscarriage, stillbirth, or death'\n\n"
        "**EXAMPLE 2:**\n"
        "If the table text is:\n"
        "'''\nAdverse outcomes at <34 wk of gestation\n"
        "  Any — no. (%)                            32 (4.0)\n"
        "  Preeclampsia — no. (%)                   3 (0.4)\n'''\n"
        "Extract as:\n"
        "- Domain: 'Adverse outcomes at <34 wk of gestation', Specific: 'Any'\n"
        "- Domain: 'Adverse outcomes at <34 wk of gestation', Specific: 'Preeclampsia'\n\n"
        "--- END OF EXAMPLES ---\n\n"
        "**OUTPUT FORMAT:** Return a JSON object with a list called 'table_outcomes'. Each item must have 'outcome_domain' and 'outcome_specific'.\n\n"
        f"**TABLE TEXT TO PARSE:**\n{table_text}"
    )
    # This agent does not need a spinner, it's part of a larger process
    response = ask_llm(prompt)
    return parse_json_response(response, "table_outcomes") or []


def agent_synthesize_and_verify(defined_outcomes: list, table_outcomes: list) -> list:
    """Agent 4: Merges all outcomes into a final, complete list."""
    prompt = (
        "You are a senior clinical data reviewer. Your goal is to create one final, complete, and deduplicated list of outcomes from the sources provided.\n"
        "1. `defined_outcomes`: Outcomes defined in the text.\n"
        "2. `table_outcomes`: Outcomes parsed from tables.\n\n"
        "Combine these lists. The `table_outcomes` list is often the most detailed and structured, so prioritize its hierarchy. "
        "Match the definitions from `defined_outcomes` to the corresponding items from `table_outcomes`.\n\n"
        "Return a final JSON object with a key 'final_outcomes'. Each item must have keys: "
        "'outcome_type', 'outcome_domain', 'outcome_specific', 'definition', 'measurement_method', 'timepoint'.\n"
        "Create 'domain' type entries for each unique domain, and 'specific' type entries for each specific outcome.\n\n"
        f"defined_outcomes = {json.dumps(defined_outcomes)}\n\n"
        f"table_outcomes = {json.dumps(table_outcomes)}"
    )
    st.write("↳ Final Agent: Synthesizing all outcomes...")
    response = ask_llm(prompt)
    return parse_json_response(response, "final_outcomes") or []


# ---------- 3. MAIN ORCHESTRATION PIPELINE ----------

def run_extraction_pipeline(file):
    """Orchestrates the entire multi-agent extraction process."""
    full_text = pdf_to_text(file)
    if not full_text:
        return None, None

    # Agent 1: Extract Metadata
    study_info = agent_extract_metadata(full_text)

    # Agent 2: Locate defined outcomes from prose
    defined_outcomes = agent_locate_defined_outcomes(full_text)

    # Agent 3: Find and parse all tables
    st.write("↳ Finding and parsing tables...")
    table_texts = re.findall(r"(Table \d+\..*?)(?=\nTable \d+\.|\Z)", full_text, re.DOTALL)
    all_table_outcomes = []
    if table_texts:
        st.success(f"✓ Found {len(table_texts)} tables to parse.")
        for i, table_text in enumerate(table_texts):
            st.write(f"  Parsing Table {i+1}...")
            parsed_outcomes = agent_parse_table(table_text)
            if parsed_outcomes:
                all_table_outcomes.extend(parsed_outcomes)
    else:
        st.warning("No tables found to parse.")

    # Agent 4: Synthesize everything
    final_outcomes = agent_synthesize_and_verify(defined_outcomes, all_table_outcomes)

    return study_info, final_outcomes


# ---------- 4. STREAMLIT UI ----------

st.set_page_config(layout="wide")
st.title("Advanced Clinical Trial Outcome Extractor")
st.markdown("This tool uses a multi-agent AI workflow with a dedicated table parser to accurately extract outcomes.")

file = st.file_uploader("Upload a PDF clinical trial report", type="pdf")

if file:
    with st.status(f"Processing {file.name}...", expanded=True) as status:
        study_info, outcomes = run_extraction_pipeline(file)

        if outcomes:
            status.update(label="Processing complete!", state="complete", expanded=False)
            df = pd.DataFrame(outcomes)
            
            final_rows = []
            if not study_info: study_info = {}
            study_info["pdf_name"] = file.name

            for outcome in outcomes:
                row = study_info.copy()
                row.update(outcome)
                final_rows.append(row)
            final_df = pd.DataFrame(final_rows)

            st.success(f"Successfully extracted {len(df['outcome_domain'].unique())} domains and {len(df[df['outcome_type'] == 'specific'])} specific outcomes.")
            st.subheader("Structured Outcome View")
            st.dataframe(df[['outcome_type', 'outcome_domain', 'outcome_specific', 'definition', 'timepoint']], use_container_width=True, hide_index=True)

            st.subheader("Export Results")
            st.download_button(
                "Download Extracted Data as CSV",
                final_df.to_csv(index=False).encode('utf-8'),
                f"extracted_outcomes_{file.name}.csv",
                "text/csv",
                key='download-csv'
            )
            
            st.subheader("Full Data Table")
            st.dataframe(final_df)
            
            st.subheader("Extracted Study Information")
            st.json(study_info)

        else:
            status.update(label="Extraction Failed", state="error", expanded=True)
            st.error("Could not extract any outcomes. The document may be unreadable or contain no recognized outcome patterns.")