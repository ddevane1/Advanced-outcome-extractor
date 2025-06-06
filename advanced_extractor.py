#!/usr/bin/env python3
# -------- advanced_extractor.py (v2.1 - final fix) --------

import os
import json
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
            # Check if there is any extractable text
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

def agent_map_document_sections(full_text: str) -> dict:
    """Agent 1: Scans the text to identify the boundaries of key sections."""
    prompt = (
        "You are a document analysis expert. Your task is to identify the key sections of this clinical trial paper. "
        "Find the start of the 'Abstract', 'Methods' (or similar like 'Patients and Methods'), and 'Results' sections. "
        "Also identify the start of the 'Tables' section if present.\n\n"
        "For each section, provide the first ~10 words to uniquely identify its beginning.\n\n"
        "Return a JSON object with keys 'abstract_start', 'methods_start', 'results_start', 'tables_start'. "
        "If a section is not found, its value should be null.\n\n"
        f"Document Text:\n{full_text[:12000]}"
    )
    st.write("↳ Agent 1: Mapping document structure...")
    response = ask_llm(prompt)
    section_map = parse_json_response(response, None)
    if section_map and any(section_map.values()):
        st.success("✓ Document sections mapped successfully.")
        return section_map
    else:
        st.warning("✗ Failed to map document sections. The document may have a non-standard format.")
        return None

def agent_extract_metadata(text_chunk: str) -> dict:
    """Agent 2: Extracts the high-level study metadata."""
    prompt = (
        "You are a metadata extraction specialist. From the provided text, "
        "extract the following study information precisely. If information is absent, use null.\n"
        "Respond in this exact JSON format:\n"
        '{"study_info": {"first_author_surname": "...", "publication_year": "...", "journal": "...", "study_design": "...", "study_country": "...", "patient_population": "...", "targeted_condition": "...", "diagnostic_criteria": "...", "interventions_tested": "...", "comparison_group": "..."}}\n\n'
        f"Text to analyze:\n{text_chunk}"
    )
    st.write("↳ Agent 2: Extracting study metadata...")
    response = ask_llm(prompt)
    return parse_json_response(response, "study_info")

def agent_locate_defined_outcomes(methods_text: str) -> list:
    """Agent 3: Finds the "planned" outcomes as defined in the Methods section."""
    prompt = (
        "You are a clinical trial protocol analyst. Your task is to extract all outcome definitions from the provided Methods section. "
        "Capture the primary and secondary outcomes, including their full definitions, measurement instruments/methods, and timepoints. "
        "Return a JSON object with a list called 'defined_outcomes'.\n\n"
        f"Methods Section Text:\n{methods_text}"
    )
    st.write("↳ Agent 3: Locating defined outcomes in Methods...")
    response = ask_llm(prompt)
    return parse_json_response(response, "defined_outcomes") or []

def agent_extract_reported_results(results_text: str) -> list:
    """Agent 4: Extracts the "reported" outcomes from the Results and Tables."""
    prompt = (
        "You are a clinical trial results analyst. Your task is to extract all reported outcomes from the provided Results section and Tables. "
        "Structure the outcomes hierarchically with a domain and specific measures. "
        "Return a JSON object with a list called 'reported_results'.\n\n"
        f"Results and Tables Text:\n{results_text}"
    )
    st.write("↳ Agent 4: Extracting reported results...")
    response = ask_llm(prompt)
    return parse_json_response(response, "reported_results") or []

def agent_synthesize_and_verify(defined_outcomes: list, reported_results: list, full_text: str) -> list:
    """Agent 5: Merges, deduplicates, and verifies the final list of outcomes."""
    prompt = (
        "You are a senior clinical data reviewer. Your task is to synthesize and verify clinical trial outcomes. "
        "You have two lists: 1. `planned_outcomes` (from Methods) and 2. `reported_results` (from Results/Tables). "
        "Your goal is to create a single, complete, and deduplicated list. For each planned outcome, find its corresponding result and merge the information. "
        "If a planned outcome seems missing from the results, re-scan the `full_document_text` to find it. "
        "Return a final JSON object with a key 'final_outcomes'. Each item must have keys: "
        "'outcome_type', 'outcome_domain', 'outcome_specific', 'definition', 'measurement_method', 'timepoint'.\n\n"
        f"planned_outcomes = {json.dumps(defined_outcomes)}\n\n"
        f"reported_results = {json.dumps(reported_results)}\n\n"
        f"full_document_text = {full_text}"
    )
    st.write("↳ Agent 5: Synthesizing and verifying final outcome list...")
    response = ask_llm(prompt)
    return parse_json_response(response, "final_outcomes") or []


# ---------- 3. FALLBACK STRATEGY ----------

def run_simple_extraction(full_text: str):
    """
    A simpler, single-agent fallback if the advanced pipeline fails.
    This prompt asks the model to do everything in one go.
    """
    st.warning("Switching to simple, single-pass extraction mode.")
    prompt = (
        "You are an expert medical reviewer extracting clinical trial data. The document could not be mapped, so analyze the entire text provided. "
        "Extract study metadata and all hierarchical outcomes (primary, secondary, adverse events). "
        "Return a single JSON object with two top-level keys: 'study_info' and 'outcomes'.\n\n"
        "'study_info' should contain: first_author_surname, publication_year, journal, study_design, etc.\n"
        "'outcomes' should be a list where each item has keys: outcome_type, outcome_domain, outcome_specific, definition, measurement_method, timepoint.\n\n"
        f"Full document text:\n{full_text}"
    )
    response = ask_llm(prompt)
    data = parse_json_response(response, None)
    if not data:
        return None, None
    return data.get("study_info"), data.get("outcomes")


# ---------- 4. MAIN ORCHESTRATION PIPELINE ----------

def run_extraction_pipeline(file):
    """
    Orchestrates the entire multi-agent extraction process with a fallback.
    """
    full_text = pdf_to_text(file)
    if not full_text:
        return None, None

    # Step 1: Try to map the document
    section_map = agent_map_document_sections(full_text)

    # If mapping fails, use the simple fallback method
    if not section_map:
        return run_simple_extraction(full_text)

    # --- If mapping succeeds, proceed with the advanced multi-agent pipeline ---
    st.success("✓ Proceeding with advanced multi-agent extraction.")

    def get_section_text(start_key, end_key=None):
        start_marker = section_map.get(start_key)
        if not start_marker: return ""
        start_index = full_text.find(start_marker)
        if start_index == -1: return ""
        end_index = len(full_text)
        if end_key and section_map.get(end_key):
            next_marker = section_map.get(end_key)
            end_index = full_text.find(next_marker, start_index)
            if end_index == -1: end_index = len(full_text)
        return full_text[start_index:end_index]

    abstract_text = get_section_text('abstract_start', 'methods_start')
    methods_text = get_section_text('methods_start', 'results_start')
    results_and_tables_text = get_section_text('results_start')

    study_info = agent_extract_metadata(abstract_text + "\n\n" + methods_text)
    defined_outcomes = agent_locate_defined_outcomes(methods_text)
    reported_results = agent_extract_reported_results(results_and_tables_text)
    final_outcomes = agent_synthesize_and_verify(defined_outcomes, reported_results, full_text)

    return study_info, final_outcomes


# ---------- 5. STREAMLIT UI ----------

st.set_page_config(layout="wide")
st.title("Advanced Clinical Trial Outcome Extractor")
st.markdown("This tool uses a multi-agent AI workflow to accurately extract and verify outcomes from PDF trial reports.")

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

            st.success(f"Successfully extracted {len(df[df['outcome_type'] == 'domain'])} domains and {len(df[df['outcome_type'] == 'specific'])} specific outcomes.")
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
            with st.expander("Show Full Data Table"):
                st.dataframe(final_df)
            with st.expander("Show Extracted Study Information"):
                st.json(study_info)

        else:
            status.update(label="Extraction Failed", state="error", expanded=True)
            st.error("Could not extract any outcomes even with the fallback method. The document is likely unreadable.")