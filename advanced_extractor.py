#!/usr/bin/env python3
# -------- v10.0 - Final Verified Application --------

import os
import json
import re
import streamlit as st
import pandas as pd
import pdfplumber
import io
from openai import OpenAI

# ----- CONFIG -----
MODEL = "gpt-4o"
DEFAULT_TOKENS_FOR_RESPONSE = 4096
LARGE_TOKENS_FOR_RESPONSE = 8192
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))

# ---------- 1. CORE HELPER FUNCTIONS ----------

@st.cache_data
def get_pdf_text(file_contents):
    """Extracts text from the bytes of an uploaded PDF file and caches the result."""
    st.info("Step 1: Reading PDF text...")
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

def ask_llm(prompt: str, is_json: bool = True, max_response_tokens: int = DEFAULT_TOKENS_FOR_RESPONSE) -> str:
    """Generic function to call the OpenAI API."""
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
    """Safely parses JSON from the LLM response."""
    if not response_text: return None
    try:
        json_str = response_text.strip().removeprefix("```json").removesuffix("```")
        data = json.loads(json_str)
        return data if key is None else data.get(key)
    except (json.JSONDecodeError, AttributeError):
        st.warning(f"Could not parse a valid JSON response from the AI.")
        return None


# ---------- 2. SPECIALIZED AGENT FUNCTIONS ----------

def agent_extract_metadata(full_text: str) -> dict:
    """Agent 1: Extracts the high-level study metadata."""
    prompt = (
        "You are a metadata extraction specialist. From the beginning of this document, extract the study information. If a value is absent, use null.\n"
        'Respond in this exact JSON format: {"study_info": {"first_author_surname": "...", "publication_year": "...", "journal": "...", "study_design": "...", "study_country": "...", "patient_population": "...", "targeted_condition": "...", "diagnostic_criteria": "...", "interventions_tested": "...", "comparison_group": "..."}}\n\n'
        f"Text to analyze:\n{full_text[:8000]}"
    )
    return parse_json_response(ask_llm(prompt), "study_info")

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
    return parse_json_response(ask_llm(prompt), "defined_outcomes") or []

def agent_parse_table(table_text: str) -> list:
    """Agent 3: A specialist agent to parse a single table."""
    prompt = (
        "You are an expert at parsing clinical trial tables. Analyze the single table text below.\n\n"
        "**STEP 1: CLASSIFY THE TABLE**\n"
        "First, determine if this table describes **baseline patient characteristics** (demographics, age, etc.) or **clinical trial outcomes** (results, events, complications).\n\n"
        "**STEP 2: EXTRACT BASED ON CLASSIFICATION**\n"
        "-   **If BASELINE table**, you MUST return an empty list: `{\"table_outcomes\": []}`\n"
        "-   **If, and ONLY IF, it is an OUTCOME table**, proceed with the extraction rules below.\n\n"
        "**CLINICAL OUTCOME TABLE PARSING RULES:**\n"
        "1.  **Extract Clean Names:** The outcome name is the text description ONLY. You **MUST STRIP AWAY** all trailing data, numbers, percentages, and formatting like '— no. (%)'. For example, if the line is 'Preeclampsia — no. (%) ... 3 (0.4)', the outcome name is just 'Preeclampsia'.\n"
        "2.  **Identify the Domain & Specific Outcomes:** The main heading is the 'outcome_domain'. Items indented under it are 'outcome_specific' measures.\n"
        "**OUTPUT FORMAT:** Return a JSON object with a list called 'table_outcomes'.\n\n"
        f"**TABLE TEXT TO PARSE:**\n{table_text}"
    )
    return parse_json_response(ask_llm(prompt), "table_outcomes") or []

def agent_finalize_and_structure(messy_list: list) -> list:
    """Agent 4: Takes a messy list of outcomes and cleans, deduplicates, and structures it."""
    prompt = (
        "You are a data structuring expert. Clean, deduplicate, and structure this messy list of outcomes into a final hierarchical list.\n\n"
        "**RULES:**\n"
        "1.  For each unique outcome domain, create one entry with `\"outcome_type\": \"domain\"`.\n"
        "2.  For each specific outcome under a domain, create an entry with `\"outcome_type\": \"specific\"`.\n"
        "3.  Combine information. If you see the same outcome multiple times, merge any definitions or timepoints.\n"
        "4.  Remove any obvious non-outcome entries or duplicates.\n"
        "**OUTPUT FORMAT:** Return a final JSON object with a key 'final_outcomes'. Each item must have keys: 'outcome_type', 'outcome_domain', 'outcome_specific', 'definition', 'measurement_method', 'timepoint'.\n\n"
        f"**MESSY LIST TO PROCESS:**\n{json.dumps(messy_list, indent=2)}"
    )
    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "final_outcomes") or []


# ---------- 3. MAIN ORCHESTRATION PIPELINE (CACHED) ----------

@st.cache_data(show_spinner="Step 2: Running AI extraction pipeline...")
def run_extraction_pipeline(full_text: str):
    """Orchestrates the AI agent calls. This entire function is cached."""
    
    study_info = agent_extract_metadata(full_text)
    defined_outcomes = agent_locate_defined_outcomes(full_text)
    
    table_texts = re.findall(r"(Table \d+\..*?)(?=\nTable \d+\.|\Z)", full_text, re.DOTALL)
    all_table_outcomes = []
    if table_texts:
        for table_text in table_texts:
            parsed_outcomes = agent_parse_table(table_text)
            if parsed_outcomes:
                all_table_outcomes.extend(parsed_outcomes)
    
    raw_combined_list = defined_outcomes + all_table_outcomes
    if not raw_combined_list:
        return study_info, []

    final_outcomes = agent_finalize_and_structure(raw_combined_list)
    
    return study_info, final_outcomes


# ---------- 4. STREAMLIT UI ----------

st.set_page_config(layout="wide")
st.title("Clinical Trial Outcome Extractor (v10.0)")
st.markdown("This tool uses a cached, multi-agent AI workflow to accurately and reliably extract outcomes.")

uploaded_file = st.file_uploader("Upload a PDF clinical trial report to begin", type="pdf")

if uploaded_file is not None:
    file_contents = uploaded_file.getvalue()
    
    # Step 1: Get text from the PDF. This is cached and runs first.
    full_text = get_pdf_text(file_contents)
    
    # Step 2: If text exists, run the main pipeline. This is also cached.
    if full_text:
        study_info, outcomes = run_extraction_pipeline(full_text)

        if outcomes:
            st.success(f"Processing complete for **{uploaded_file.name}**.")
            df = pd.DataFrame(outcomes)
            for col in ['outcome_domain', 'outcome_specific', 'outcome_type', 'definition', 'timepoint']:
                if col not in df.columns: df[col] = ''
            df.fillna('', inplace=True)

            # HIERARCHICAL DISPLAY
            st.subheader("Hierarchical Outcome View")
            domains = df[df['outcome_domain'] != '']['outcome_domain'].unique()
            for domain in domains:
                st.markdown(f"**DOMAIN:** {domain}")
                specific_outcomes = df[(df['outcome_domain'] == domain) & (df['outcome_specific'] != '') & (df['outcome_specific'] != domain)]['outcome_specific'].unique()
                if len(specific_outcomes) > 0:
                    for specific in specific_outcomes:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {specific}")
                else:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• *This is a primary outcome or a domain with no specific sub-outcomes listed.*")
                st.write("") 

            # DATA EXPORT
            st.subheader("Export Results")

            # Simplified CSV
            simplified_rows = []
            for domain in domains:
                domain_row = df[df['outcome_domain'] == domain].iloc[0]
                simplified_rows.append({"Level": "DOMAIN", "Outcome": domain, "Timepoint": domain_row.get('timepoint', '')})
                specific_outcomes_df = df[(df['outcome_domain'] == domain) & (df['outcome_specific'] != '') & (df['outcome_specific'] != domain)]
                for _, specific_row in specific_outcomes_df.iterrows():
                    simplified_rows.append({"Level": "Specific", "Outcome": f"  • {specific_row['outcome_specific']}", "Timepoint": specific_row.get('timepoint', '')})
            simplified_df = pd.DataFrame(simplified_rows)

            st.download_button(
                label="**Download Simplified View (Recommended)**",
                data=simplified_df.to_csv(index=False).encode('utf-8'),
                file_name=f"Simplified_Outcomes_{uploaded_file.name}.csv",
                mime='text/csv',
                help="A clean, human-readable list of the outcome hierarchy."
            )

            # Full Data Table (in expander)
            with st.expander("Show Advanced Export & Full Data"):
                final_rows = []
                if study_info: study_info["pdf_name"] = uploaded_file.name
                for _, row in df.iterrows():
                    new_row = study_info.copy() if study_info else {}
                    new_row.update(row.to_dict())
                    final_rows.append(new_row)
                final_df = pd.DataFrame(final_rows)

                st.download_button(
                    label="Download Full Raw Data",
                    data=final_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"Full_Data_{uploaded_file.name}.csv",
                    mime='text/csv',
                    help="The complete raw data, useful for computer analysis."
                )
                st.markdown("**Full Raw Data Table:**")
                st.dataframe(final_df)
                st.markdown("**Extracted Study Information:**")
                st.json(study_info or {})
        else:
            st.error("Extraction ran but no outcomes were found. The document may not contain recognizable outcome patterns.")