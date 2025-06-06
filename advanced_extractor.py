#!/usr/bin/env python3
# -------- advanced_extractor.py (v8.1 - final syntax verification) --------

import os
import json
import re
import streamlit as st
import pandas as pd
import pdfplumber
import tiktoken
from openai import OpenAI
import io

# ----- CONFIG -----
MODEL = "gpt-4o"
DEFAULT_TOKENS_FOR_RESPONSE = 4096
LARGE_TOKENS_FOR_RESPONSE = 8192
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))

# ---------- 1. CORE HELPER FUNCTIONS ----------

@st.cache_data
def get_pdf_text(file_contents):
    """Extracts text from the bytes of an uploaded PDF file and caches the result."""
    st.info("Reading PDF text...")
    try:
        with pdfplumber.open(io.BytesIO(file_contents)) as pdf:
            full_text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
            if not full_text.strip():
                st.error("This PDF appears to be a scanned image or contains no extractable text.")
                return None
            st.success("PDF text read successfully.")
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
        st.warning(f"Could not parse a valid JSON response. The AI may have failed to generate a complete answer.")
        return None


# ---------- 2. SPECIALIZED AGENT FUNCTIONS ----------

def agent_extract_metadata(full_text: str) -> dict:
    """Agent 1: Extracts the high-level study metadata."""
    prompt = (
        "You are a metadata extraction specialist. From the beginning of this document, extract study information. If absent, use null.\n"
        'Respond in JSON: {"study_info": {"first_author_surname": "...", "publication_year": "...", "journal": "...", "study_design": "...", "study_country": "...", "patient_population": "...", "targeted_condition": "...", "diagnostic_criteria": "...", "interventions_tested": "...", "comparison_group": "..."}}\n\n'
        f"Text to analyze:\n{full_text[:6000]}"
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
        "1.  **Extract Clean Names:** The outcome name is the text description ONLY. You **MUST STRIP AWAY** all trailing data, numbers, percentages, and formatting like '— no. (%)'.\n"
        "2.  **Identify the Domain & Specific Outcomes:** The main heading is the 'outcome_domain'. Items indented under it are 'outcome_specific' measures.\n"
        "**OUTPUT FORMAT:** Return a JSON object with a list called 'table_outcomes'.\n\n"
        f"**TABLE TEXT TO PARSE:**\n{table_text}"
    )
    return parse_json_response(ask_llm(prompt), "