#!/usr/bin/env python3
# -------- advanced_extractor.py (v2.2 - final UI fix) --------

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
        "Return a JSON object with keys 'abstract_start', 'methods_start', 'results_start', 'tables_start'. If a section is not found, use null.\n\n"
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
        "You are a metadata extraction specialist. From the provided text, extract study information precisely. If absent, use null.\n"
        'Respond in JSON: {"study_info": {"first_author_surname": "...", "publication_year": "...", "journal": "...", "study_design": "...", "study_country": "...", "patient_population": "...", "targeted_condition": "...", "diagnostic_criteria": "...", "interventions_tested": "...", "comparison_group": "..."}}\n\n'
        f"Text to analyze:\n{text_chunk}"
    )
    st.write("↳ Agent 2: Extracting study metadata...")
    response = ask_llm(prompt)
    return parse_json_response(response, "study_info")

def agent_locate_defined_outcomes(methods_text: str) -> list:
    """Agent 3: Finds the "planned" outcomes as defined in the Methods section."""
    prompt = (
        "You are a clinical trial protocol analyst. Extract all outcome definitions from the provided Methods section. "
        "Capture primary/secondary outcomes, definitions, measurement instruments, and timepoints. "
        "Return a JSON object with a list called 'defined_outcomes'.\n\n"
        f"Methods Section Text:\n{methods_text}"
    )
    st.write("↳ Agent 3: Locating defined outcomes in Methods...")
    response = ask_llm(prompt)
    return parse_json_response(response, "defined_outcomes") or []

def agent_extract_reported_results(results_text: str) -> list:
    """Agent 4: Extracts the "reported" outcomes from the Results and Tables."""
    prompt = (
        "You are a clinical trial results analyst. Extract all reported outcomes from the provided Results section and Tables. "