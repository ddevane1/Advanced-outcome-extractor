#!/usr/bin/env python3
# -------- advanced_extractor.py (v4.0 - dedicated table parser) --------

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

def agent_map_document_sections(full_text: str) -> dict:
    """Agent 1: Scans the text to identify key sections."""
    st.write("↳ Agent 1: Mapping document structure...")
    # This agent's implementation can be simplified by assuming a standard structure
    # or enhanced with more complex logic if needed. For now, we'll pass.
    return {"status": "success"} # Assume we'll search the whole text for now

def agent_extract_metadata(full_text: str) -> dict:
    """Agent 2: Extracts the high-level study metadata from the first few pages."""
    prompt = (
        "You are a metadata extraction specialist. From the beginning of this document, extract study information. If absent, use null.\n"
        'Respond in JSON: {"study_info": {"first_author_surname": "...", "publication_year": "...", "journal": "...", "study_design": "...", "study_country": "...", "patient_population": "...", "targeted_condition": "...", "diagnostic_criteria": "...", "interventions_tested": "...", "comparison_group": "..."}}\n\n'
        f"Text to analyze:\n{full_text[:6000]}"
    )
    st.write("↳ Agent 2: Extracting study metadata...")
    response = ask_llm(prompt)
    return parse_json_response(response, "study_info")

def agent_locate_defined_outcomes(full_text: str) -> list:
    """Agent 3: Finds planned outcomes from the Methods section using specific rules."""
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
    st.write("↳ Agent 3: Locating defined outcomes from text...")
    response = ask_llm(prompt)
    return parse_json_response(response, "defined_outcomes") or []


def agent_parse_table(table_text: str) -> list:
    """Agent 4: A specialist agent to parse the text of a single table."""
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
    """Agent 5: Merges all outcomes into a final