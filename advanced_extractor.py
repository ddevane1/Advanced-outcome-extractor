#!/usr/bin/env python3
# -------- advanced_extractor.py (v5.0 - final polishing) --------

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
DEFAULT_TOKENS_FOR_RESPONSE = 4096
LARGE_TOKENS_FOR_RESPONSE = 8192
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))

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

def ask_llm(prompt: str, is_json: bool = True, max_response_tokens: int = DEFAULT_TOKENS_FOR_RESPONSE) -> str:
    """Generic function to call the OpenAI API, with adjustable token limit."""
    try:
        response_format = {"type": "json_object"} if is_json else {"type": "text"}
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_response_tokens,
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
        st.warning(f"Could not parse JSON for key '{key}'. The response may have been cut off. Raw response: {response_text[:500]}")
        return None


# ---------- 2. SPECIALIZED AGENT FUNCTIONS ----------

def agent_extract_metadata(full_text: str) -> dict:
    """Agent 1: Extracts the high-level study metadata."""
    prompt = (
        "You are a metadata extraction specialist. From the beginning of this document, extract study information. If absent, use null.\n"
        'Respond in JSON: {"study_info": {...}}\n\n'
        f"Text to analyze:\n{full_text[:6000]}"
    )
    st.write("↳ Agent 1: Extracting study metadata...")
    response = ask_llm(prompt)
    return parse_json_response(response, "study_info")

def agent_locate_defined_outcomes(full_text: str) -> list:
    """Agent 2: Finds planned outcomes from the Methods section."""
    prompt = (
        "You are a clinical trial protocol analyst. Extract all outcome definitions, typically found in the 'Methods' section.\n\n"
        "**RULES:**\n"
        "1.  **Handle Semicolon-Separated Lists:** Treat each item in a semicolon-separated list as a separate outcome domain.\n"
        "2.  **Handle Time-Based Grouping:** Create a separate domain for each timepoint (e.g., 'before 34 weeks').\n\n"
        "**OUTPUT FORMAT:** Return a JSON object with a list called 'defined_outcomes'.\n\n"
        f"**Document Text to Analyze:**\n{full_text}"
    )
    st.write("↳ Agent 2: Locating defined outcomes from text...")
    response = ask_llm(prompt)
    return parse_json_response(response, "defined_outcomes") or []


def agent_parse_table(table_text: str) -> list:
    """Agent 3: A specialist agent to parse a single table, ignoring baseline tables and cleaning outcome names."""
    prompt = (
        "You are an expert at parsing clinical trial tables. Analyze the single table text below.\n\n"
        "**STEP 1: CLASSIFY THE TABLE**\n"
        "First, determine if this table describes **baseline patient characteristics** (demographics, age, etc.) or **clinical trial outcomes** (results, events, complications).\n\n"
        "**STEP 2: EXTRACT BASED ON CLASSIFICATION**\n"
        "-   **If BASELINE table**, you MUST return an empty list: `{\"table_outcomes\": []}`\n"
        "-   **If, and ONLY IF, it is an OUTCOME table**, proceed with the rules below.\n\n"
        "**CLINICAL OUTCOME TABLE PARSING RULES:**\n"
        "1.  **Extract Clean Names:** The outcome name is the text description ONLY. You **MUST STRIP AWAY** all trailing data, numbers, percentages, and formatting like '— no. (%)'. For example, if the line is 'Preeclampsia — no. (%) ... 3 (0.4)', the outcome name is just 'Preeclampsia'.\n"
        "2.  **Identify the Domain:** The main heading for a group of outcomes is the 'outcome_domain'.\n"
        "3.  **Identify Specific Outcomes:** Items indented under a domain are the 'outcome_specific' measures.\n"
        "4.  **Handle Primary Outcomes:** If a line looks like a primary outcome (e.g., 'Primary outcome: preterm preeclampsia'), treat the text *after* the colon as both the domain and the specific outcome, but ensure it is cleaned per Rule #1.\n\n"
        "**OUTPUT FORMAT:** Return a JSON object with a list called 'table_outcomes'. This list will be empty for baseline tables.\n\n"
        f"**TABLE TEXT TO PARSE:**\n{table_text}"
    )
    response = ask_llm(prompt)
    return parse_json_response(response, "table_outcomes") or []


def agent_synthesize_and_verify(defined_outcomes: list, table_outcomes: list) -> list:
    """Agent 4: Merges all outcomes into a final, complete list."""
    prompt = (
        "You are a senior clinical data reviewer. Your goal is to create one final, complete, and deduplicated list of outcomes from the sources provided.\n"
        "1. `defined_outcomes`: Outcomes defined in the text.\n"
        "2. `table_outcomes`: Cleaned outcomes parsed from tables.\n\n"
        "Combine these lists, prioritizing the hierarchy from the `table_outcomes`. Match definitions from `defined_outcomes` to the corresponding items. "
        "Return a JSON object with a key 'final_outcomes'. Each item must have keys: "
        "'outcome_type', 'outcome_domain', 'outcome_specific', 'definition', 'measurement_method', 'timepoint'.\n"
        "Create 'domain' type entries and 'specific' type entries.\n\n"
        f"defined_outcomes = {json.dumps(defined_outcomes)}\n\n"
        f"table_outcomes = {json.dumps(table_outcomes)}"
    )
    st.write("↳ Final Agent: Synthesizing all outcomes...")
    response = ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE)
    return parse_json_response(response, "final_outcomes") or []


# ---------- 3. MAIN ORCHESTRATION PIPELINE ----------

def run_extraction_pipeline(file):
    """Orchestrates the entire multi-agent extraction process."""
    full_text = pdf_to_text(file)
    if not full_text:
        return None, None

    study_info = agent_extract_metadata(full_text)
    defined_outcomes = agent_locate_defined_outcomes(full_text)

    st.write("↳ Finding and parsing tables...")
    table_texts = re.findall(r"(Table \d+\..*?)(?=\nTable \d+\.|\Z)", full_text, re.DOTALL)
    all_table_outcomes = []
    if table_texts:
        st.success(f"✓ Found {len(table_texts)} tables to parse.")
        for i, table_text in enumerate(table_texts):
            st.write(f"  Analyzing Table {i+1}...")
            parsed_outcomes = agent_parse_table(table_text)
            if parsed_outcomes:
                st.write(f"    -> Table {i+1} is an OUTCOME table. Extracted {len(parsed_outcomes)} items.")
                all_table_outcomes.extend(parsed_outcomes)
            else:
                st.write(f"    -> Table {i+1} is a BASELINE table. Skipping.")
    else:
        st.warning("No tables found to parse.")

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
            # Ensure we have the necessary columns, fill with empty string if not
            for col in ['outcome_domain', 'outcome_specific', 'outcome_type']:
                if col not in df.columns:
                    df[col] = ''
            df.fillna('', inplace=True)

            # --- NEW HIERARCHICAL DISPLAY ---
            st.subheader("Hierarchical Outcome View")
            domains = df[df['outcome_domain'] != '']['outcome_domain'].unique()

            for domain in domains:
                st.markdown(f"**DOMAIN:** {domain}")
                # Get specific outcomes for this domain
                specific_outcomes = df[
                    (df['outcome_domain'] == domain) & 
                    (df['outcome_specific'] != '') &
                    # Avoid printing the domain name as its own specific outcome
                    (df['outcome_specific'] != domain) 
                ]['outcome_specific'].unique()

                if len(specific_outcomes) > 0:
                    for specific in specific_outcomes:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {specific}")
                else:
                    # Handle cases where a domain is listed but has no specific sub-outcomes in the final data
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• *No specific outcomes listed under this domain.*")
                st.write("") # Add a little space

            # --- DATA EXPORT & FULL TABLE ---
            final_rows = []
            if not study_info: study_info = {}
            study_info["pdf_name"] = file.name
            for _, row in df.iterrows():
                new_row = study_info.copy()
                new_row.update(row.to_dict())
                final_rows.append(new_row)
            final_df = pd.DataFrame(final_rows)

            st.subheader("Export Results")
            st.download_button(
                "Download Extracted Data as CSV",
                final_df.to_csv(index=False).encode('utf-8'),
                f"extracted_outcomes_{file.name}.csv",
                "text/csv",
                key='download-csv'
            )
            
            with st.expander("Show Full Data Table"):
                st.dataframe(final_df)
            
            with st.expander("Show Extracted Study Information"):
                st.json(study_info)

        else:
            status.update(label="Extraction Failed", state="error", expanded=True)
            st.error("Could not extract any outcomes. The document may be unreadable or contain no recognized outcome patterns.")