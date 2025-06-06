#!/usr/bin/env python3
# -------- advanced_extractor.py (multi-agent strategy) --------

import os
import json
import streamlit as st
import pandas as pd
import pdfplumber
import tiktoken
from openai import OpenAI

# ----- CONFIG -----
MODEL = "gpt-4o"
TOKENS_FOR_RESPONSE = 4000  # Increased for potentially larger JSON objects
# It's recommended to set the API key in your environment variables
# For Streamlit Cloud, set this in the app's secrets management
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))
enc = tiktoken.encoding_for_model(MODEL)

# ---------- 1. CORE HELPER FUNCTIONS ----------

def pdf_to_text(file):
    """Extracts text from an uploaded PDF file."""
    try:
        with pdfplumber.open(file) as pdf:
            return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
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
        # The model sometimes wraps the JSON in markdown, so we strip it
        json_str = response_text.strip().removeprefix("```json").removesuffix("```")
        return json.loads(json_str).get(key)
    except (json.JSONDecodeError, AttributeError) as e:
        st.warning(f"Could not parse JSON for key '{key}'. Raw response: {response_text[:500]}")
        return None


# ---------- 2. SPECIALIZED AGENT FUNCTIONS ----------

def agent_map_document_sections(full_text: str) -> dict:
    """
    Agent 1: Scans the text to identify the boundaries of key sections.
    """
    prompt = (
        "You are a document analysis expert. Your task is to identify the key sections of this clinical trial paper. "
        "Find the start of the 'Abstract', 'Methods' (or similar like 'Patients and Methods'), and 'Results' sections. "
        "Also identify the start of the 'Tables' section if present.\n\n"
        "For each section, provide the first ~10 words to uniquely identify its beginning.\n\n"
        "Return a JSON object with keys 'abstract_start', 'methods_start', 'results_start', 'tables_start'. "
        "If a section is not found, its value should be null.\n\n"
        "Example Response:\n"
        '{"abstract_start": "ABSTRACT Background Preeclampsia is a multisystem disorder...", "methods_start": "Methods Study Design and Participants This trial was an...", "results_start": "Results A total of 1805 women underwent randomization...", "tables_start": "Table 1. Characteristics of the Trial Participants."}\n\n'
        f"Document Text:\n{full_text[:12000]}" # Use the beginning of the doc for mapping
    )
    st.write("↳ Agent 1: Mapping document structure...")
    response = ask_llm(prompt)
    section_map = parse_json_response(response, None) # Get the whole dict
    if section_map:
        st.success("✓ Document sections mapped.")
    else:
        st.error("✗ Failed to map document sections.")
    return section_map

def agent_extract_metadata(text_chunk: str) -> dict:
    """
    Agent 2: Extracts the high-level study metadata.
    """
    prompt = (
        "You are a metadata extraction specialist. From the provided text (usually Abstract and Methods), "
        "extract the following study information precisely. Do not infer. If information is absent, use null.\n"
        "Respond in this exact JSON format:\n"
        "{\n"
        '  "study_info": {\n'
        '    "first_author_surname": "Author surname or None",\n'
        '    "publication_year": "Year of publication as YYYY or None",\n'
        '    "journal": "Journal name or None",\n'
        '    "study_design": "Study design or None",\n'
        '    "study_country": "Countries or None",\n'
        '    "patient_population": "Population description or None",\n'
        '    "targeted_condition": "Condition studied or None",\n'
        '    "diagnostic_criteria": "Diagnostic criteria or None",\n'
        '    "interventions_tested": "Intervention or None",\n'
        '    "comparison_group": "Comparator or None"\n'
        '  }\n'
        '}\n\n'
        f"Text to analyze:\n{text_chunk}"
    )
    st.write("↳ Agent 2: Extracting study metadata...")
    response = ask_llm(prompt)
    metadata = parse_json_response(response, "study_info")
    if metadata:
        st.success("✓ Metadata extracted.")
    else:
        st.error("✗ Failed to extract metadata.")
    return metadata

def agent_locate_defined_outcomes(methods_text: str) -> list:
    """
    Agent 3: Finds the "planned" outcomes as defined in the Methods section.
    """
    prompt = (
        "You are a clinical trial protocol analyst. Your task is to extract all outcome definitions from the provided Methods section. "
        "Capture the primary and secondary outcomes, including their full definitions, measurement instruments/methods, and timepoints. "
        "This list represents the *planned* outcomes.\n\n"
        "Return a JSON object containing a list called 'defined_outcomes'. Each item in the list should be an object with keys: "
        "'outcome_domain', 'outcome_specific', 'definition', 'measurement_method', 'timepoint'.\n\n"
        f"Methods Section Text:\n{methods_text}"
    )
    st.write("↳ Agent 3: Locating defined outcomes in Methods...")
    response = ask_llm(prompt)
    outcomes = parse_json_response(response, "defined_outcomes")
    if outcomes:
        st.success(f"✓ Found {len(outcomes)} outcome definitions in Methods.")
    else:
        st.error("✗ Failed to locate defined outcomes.")
    return outcomes or []

def agent_extract_reported_results(results_text: str) -> list:
    """
    Agent 4: Extracts the "reported" outcomes from the Results and Tables.
    """
    prompt = (
        "You are a clinical trial results analyst. Your task is to extract all reported outcomes from the provided Results section and Tables. "
        "Focus on the actual data presented. Structure the outcomes hierarchically with a domain and specific measures.\n\n"
        "Return a JSON object containing a list called 'reported_results'. Each item in the list should be an object with keys: "
        "'outcome_domain', 'outcome_specific', and 'timepoint' if mentioned directly with the result.\n\n"
        f"Results and Tables Text:\n{results_text}"
    )
    st.write("↳ Agent 4: Extracting reported results...")
    response = ask_llm(prompt)
    results = parse_json_response(response, "reported_results")
    if results:
        st.success(f"✓ Extracted {len(results)} reported results.")
    else:
        st.error("✗ Failed to extract reported results.")
    return results or []

def agent_synthesize_and_verify(defined_outcomes: list, reported_results: list, full_text: str) -> list:
    """
    Agent 5: Merges, deduplicates, and verifies the final list of outcomes.
    """
    prompt = (
        "You are a senior clinical data reviewer. Your task is to synthesize and verify clinical trial outcomes. "
        "You have received two lists:\n"
        "1. `planned_outcomes`: From the study's Methods section.\n"
        "2. `reported_results`: From the study's Results and Tables.\n\n"
        "Your goal is to create a single, complete, and deduplicated list of all outcomes. Follow these steps:\n"
        "1. For each outcome in `planned_outcomes`, find its corresponding entry in `reported_results`.\n"
        "2. Merge the information. Use the detailed 'definition' and 'measurement_method' from the planned list and combine it with the reported result.\n"
        "3. If a planned outcome seems to be completely missing from the reported results, explicitly note this and then re-scan the `full_document_text` provided below to find any mention of it. Add it to the final list with a note that its results may be missing.\n"
        "4. Add any reported results that were not in the planned list (e.g., adverse events reported in a table).\n"
        "5. Structure the final output hierarchically. For each outcome, create a 'domain' and a 'specific' measure. The primary outcome should have its own domain.\n\n"
        "Return a final, unified JSON object with a key 'final_outcomes'. Each item should have the keys: "
        "'outcome_type' ('domain' or 'specific'), 'outcome_domain', 'outcome_specific', 'definition', 'measurement_method', 'timepoint'.\n\n"
        "Data for Synthesis:\n"
        f"planned_outcomes = {json.dumps(defined_outcomes, indent=2)}\n\n"
        f"reported_results = {json.dumps(reported_results, indent=2)}\n\n"
        f"full_document_text = {full_text}"
    )
    st.write("↳ Agent 5: Synthesizing and verifying final outcome list...")
    response = ask_llm(prompt)
    final_list = parse_json_response(response, "final_outcomes")
    if final_list:
        st.success("✓ Final outcome list synthesized and verified.")
    else:
        st.error("✗ Failed to synthesize the final outcome list.")
    return final_list or []


# ---------- 3. MAIN ORCHESTRATION PIPELINE ----------

def run_extraction_pipeline(file):
    """
    Orchestrates the entire multi-agent extraction process.
    """
    full_text = pdf_to_text(file)
    if not full_text:
        return None, None

    # Step 1: Map the document
    section_map = agent_map_document_sections(full_text)
    if not section_map:
        st.error("Pipeline stopped: Could not map document sections.")
        return None, None

    # Helper to get text for a section
    def get_section_text(start_key, end_key=None):
        start_marker = section_map.get(start_key)
        if not start_marker: return ""
        start_index = full_text.find(start_marker)
        if start_index == -1: return ""
        
        # Find the end by looking for the start of the next section
        end_index = len(full_text)
        if end_key and section_map.get(end_key):
            next_marker = section_map.get(end_key)
            end_index = full_text.find(next_marker, start_index)
            if end_index == -1: end_index = len(full_text)
        
        return full_text[start_index:end_index]

    # Prepare text chunks for agents
    abstract_text = get_section_text('abstract_start', 'methods_start')
    methods_text = get_section_text('methods_start', 'results_start')
    results_and_tables_text = get_section_text('results_start') # From results to end

    # Step 2: Extract Metadata
    study_info = agent_extract_metadata(abstract_text + "\n\n" + methods_text)
    if not study_info: study_info = {} # Ensure study_info is a dict

    # Step 3: Locate Defined Outcomes
    defined_outcomes = agent_locate_defined_outcomes(methods_text)

    # Step 4: Extract Reported Results
    reported_results = agent_extract_reported_results(results_and_tables_text)

    # Step 5: Synthesize and Verify
    final_outcomes = agent_synthesize_and_verify(defined_outcomes, reported_results, full_text)

    return study_info, final_outcomes

# ---------- 4. STREAMLIT UI ----------

st.set_page_config(layout="wide")
st.title("Advanced Clinical Trial Outcome Extractor")
st.markdown("This tool uses a multi-agent AI workflow to accurately extract and verify outcomes from PDF trial reports.")

file = st.file_uploader("Upload a PDF clinical trial report", type="pdf")

if file:
    with st.status(f"Processing {file.name}...", expanded=True) as status:
        study_info, outcomes = run_extraction_pipeline(file)

        if outcomes:
            status.update(label="Processing complete!", state="complete", expanded=False)

            # --- Convert to DataFrame for Display ---
            rows = []
            if not study_info: study_info = {}
            study_info["pdf_name"] = file.name

            for outcome in outcomes:
                row = study_info.copy()
                row.update({
                    "outcome_type": outcome.get("outcome_type", ""),
                    "outcome_domain": outcome.get("outcome_domain", ""),
                    "outcome_specific": outcome.get("outcome_specific", ""),
                    "outcome_definition": outcome.get("definition", "None"),
                    "measurement_method": outcome.get("measurement_method", "None"),
                    "timepoint": outcome.get("timepoint", "None"),
                })
                rows.append(row)
            
            df = pd.DataFrame(rows)

            # --- Display Results ---
            st.success(f"Successfully extracted {len(df[df['outcome_type'] == 'domain'])} domains and {len(df[df['outcome_type'] == 'specific'])} specific outcomes.")

            st.subheader("Structured Outcome View")
            display_rows = []
            # Use unique domains from the final dataframe
            unique_domains = df.drop_duplicates(subset=['outcome_domain'])['outcome_domain'].tolist()

            for domain_name in unique_domains:
                domain_data = df[(df['outcome_domain'] == domain_name) & (df['outcome_type'] == 'domain')]
                if domain_data.empty: continue # Should not happen, but safe check

                domain_row = domain_data.iloc[0]
                display_rows.append({
                    'Level': '▼ DOMAIN',
                    'Outcome': domain_name,
                    'Definition': domain_row.get('outcome_definition', ''),
                    'Timepoint': domain_row.get('timepoint', '')
                })

                specifics = df[(df['outcome_domain'] == domain_name) & (df['outcome_type'] == 'specific')]
                for _, specific_row in specifics.iterrows():
                    display_rows.append({
                        'Level': '    → Specific',
                        'Outcome': specific_row['outcome_specific'],
                        'Definition': specific_row['outcome_definition'],
                        'Timepoint': specific_row['timepoint']
                    })

            display_df = pd.DataFrame(display_rows)[['Level', 'Outcome', 'Definition', 'Timepoint']]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # --- Download Options ---
            st.subheader("Export Results")
            st.download_button(
                "Download Extracted Data as CSV",
                df.to_csv(index=False).encode('utf-8'),
                f"extracted_outcomes_{file.name}.csv",
                "text/csv",
                key='download-csv'
            )

            # --- Detailed Data View ---
            with st.expander("Show Full Data Table"):
                st.dataframe(df)
            with st.expander("Show Extracted Study Information"):
                st.json(study_info)

        else:
            status.update(label="Extraction Failed", state="error", expanded=True)
            st.error("Could not extract any outcomes. The document might be a scanned image or have an unusual format.")