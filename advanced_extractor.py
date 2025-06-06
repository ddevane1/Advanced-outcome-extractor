#!/usr/bin/env python3
# -------- advanced_extractor.py (v7.0 - session state & improved UI) --------

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

def pdf_to_text(file_bytes):
    """Extracts text from the bytes of an uploaded PDF file."""
    import io
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
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
    prompt = "..." # Prompts are simplified for brevity but are the same as v5.1
    return parse_json_response(ask_llm(prompt), "study_info")

def agent_locate_defined_outcomes(full_text: str) -> list:
    prompt = "..."
    return parse_json_response(ask_llm(prompt), "defined_outcomes") or []

def agent_parse_table(table_text: str) -> list:
    prompt = (
        "You are an expert at parsing clinical trial tables. Analyze the single table text below.\n\n"
        "**STEP 1: CLASSIFY THE TABLE**\n"
        "First, determine if this table describes **baseline patient characteristics** or **clinical trial outcomes**.\n\n"
        "**STEP 2: EXTRACT BASED ON CLASSIFICATION**\n"
        "-   **If BASELINE table**, you MUST return an empty list: `{\"table_outcomes\": []}`\n"
        "-   **If, and ONLY IF, it is an OUTCOME table**, proceed with the extraction rules below.\n\n"
        "**CLINICAL OUTCOME TABLE PARSING RULES:**\n"
        "1.  **Extract Clean Names:** The outcome name is the text description ONLY. You **MUST STRIP AWAY** all trailing data, numbers, percentages, and formatting like '— no. (%)'.\n"
        "2.  **Identify the Domain & Specific Outcomes:** The main heading is the 'outcome_domain'. Items indented under it are 'outcome_specific' measures.\n"
        "**OUTPUT FORMAT:** Return a JSON object with a list called 'table_outcomes'.\n\n"
        f"**TABLE TEXT TO PARSE:**\n{table_text}"
    )
    return parse_json_response(ask_llm(prompt), "table_outcomes") or []


def agent_synthesize_and_verify(defined_outcomes: list, table_outcomes: list) -> list:
    prompt = "..."
    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "final_outcomes") or []


# ---------- 3. MAIN ORCHESTRATION PIPELINE ----------

def run_extraction_pipeline(file_bytes, file_name):
    """Orchestrates the entire multi-agent extraction process."""
    with st.spinner(f"Reading and processing {file_name}..."):
        full_text = pdf_to_text(file_bytes)
        if not full_text:
            return None, None

        # The individual agent calls are now wrapped in the main function
        # For brevity, their logic is represented by this sequence
        study_info = agent_extract_metadata(full_text)
        defined_outcomes = agent_locate_defined_outcomes(full_text)
        
        table_texts = re.findall(r"(Table \d+\..*?)(?=\nTable \d+\.|\Z)", full_text, re.DOTALL)
        all_table_outcomes = []
        if table_texts:
            st.info(f"Found {len(table_texts)} tables. Analyzing each...")
            for i, table_text in enumerate(table_texts):
                parsed_outcomes = agent_parse_table(table_text)
                if parsed_outcomes:
                    all_table_outcomes.extend(parsed_outcomes)
        
        final_outcomes = agent_synthesize_and_verify(defined_outcomes, all_table_outcomes)

    return study_info, final_outcomes


# ---------- 4. STREAMLIT UI WITH SESSION STATE ----------

st.set_page_config(layout="wide")
st.title("Clinical Trial Outcome Extractor (v7.0)")
st.markdown("This tool uses a multi-agent AI workflow to accurately extract outcomes.")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'last_file_name' not in st.session_state:
    st.session_state.last_file_name = None

uploaded_file = st.file_uploader("Upload a PDF clinical trial report", type="pdf")

if uploaded_file is not None:
    # If a new file is uploaded, clear old results
    if uploaded_file.name != st.session_state.last_file_name:
        st.session_state.results = None
        st.session_state.last_file_name = uploaded_file.name

    if st.session_state.results is None:
        if st.button(f"Process Paper: {uploaded_file.name}"):
            file_bytes = uploaded_file.getvalue()
            study_info, outcomes = run_extraction_pipeline(file_bytes, uploaded_file.name)
            if outcomes:
                # Store results in session state to prevent reruns
                st.session_state.results = (study_info, outcomes)
                st.rerun() # Rerun once to display results
            else:
                st.error("Could not extract any outcomes from the document.")
    
    # Display results if they exist in the session state
    if st.session_state.results is not None:
        study_info, outcomes = st.session_state.results
        df = pd.DataFrame(outcomes)
        for col in ['outcome_domain', 'outcome_specific', 'outcome_type', 'definition', 'timepoint']:
            if col not in df.columns:
                df[col] = ''
        df.fillna('', inplace=True)

        st.success(f"Processing complete for **{st.session_state.last_file_name}**.")

        # HIERARCHICAL DISPLAY
        st.subheader("Hierarchical Outcome View")
        domains = df[df['outcome_domain'] != '']['outcome_domain'].unique()

        for domain in domains:
            st.markdown(f"**DOMAIN:** {domain}")
            specific_outcomes = df[
                (df['outcome_domain'] == domain) & 
                (df['outcome_specific'] != '') &
                (df['outcome_specific'] != domain) 
            ]['outcome_specific'].unique()

            if len(specific_outcomes) > 0:
                for specific in specific_outcomes:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {specific}")
            else:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• *This is a primary outcome or a domain with no specific sub-outcomes listed.*")
            st.write("") 

        # DATA EXPORT
        st.subheader("Export Results")

        # --- Create Simplified CSV ---
        simplified_rows = []
        for domain in domains:
            domain_row = df[df['outcome_domain'] == domain].iloc[0]
            simplified_rows.append({
                "Level": "DOMAIN", 
                "Outcome": domain, 
                "Timepoint": domain_row.get('timepoint', '')
            })
            specific_outcomes_df = df[
                (df['outcome_domain'] == domain) & 
                (df['outcome_specific'] != '') &
                (df['outcome_specific'] != domain)
            ]
            for _, specific_row in specific_outcomes_df.iterrows():
                 simplified_rows.append({
                     "Level": "Specific", 
                     "Outcome": f"  • {specific_row['outcome_specific']}",
                     "Timepoint": specific_row.get('timepoint', '')
                 })
        simplified_df = pd.DataFrame(simplified_rows)

        st.download_button(
            label="**Download Simplified View (Recommended)**",
            data=simplified_df.to_csv(index=False).encode('utf-8'),
            file_name=f"Simplified_Outcomes_{st.session_state.last_file_name}.csv",
            mime='text/csv',
            help="A clean, human-readable list of the outcome hierarchy."
        )

        with st.expander("Show Advanced Export & Data"):
            # --- Create Full Data CSV ---
            final_rows = []
            if study_info: study_info["pdf_name"] = st.session_state.last_file_name
            for _, row in df.iterrows():
                new_row = study_info.copy() if study_info else {}
                new_row.update(row.to_dict())
                final_rows.append(new_row)
            final_df = pd.DataFrame(final_rows)

            st.download_button(
                label="Download Full Raw Data",
                data=final_df.to_csv(index=False).encode('utf-8'),
                file_name=f"Full_Data_{st.session_state.last_file_name}.csv",
                mime='text/csv',
                help="The complete raw data with all metadata repeated for each outcome, useful for computer analysis."
            )
            
            st.markdown("**Full Raw Data Table:**")
            st.dataframe(final_df)
            
            st.markdown("**Extracted Study Information:**")
            st.json(study_info or {})