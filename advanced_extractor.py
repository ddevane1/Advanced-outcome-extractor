#!/usr/bin/env python3
# -------- advanced_extractor.py (v21.0 – Final, Definitive Build) --------
"""
Clinical-Trial Outcome Extractor
Patch notes v21.0
- This version combines the most stable codebase with the most intelligent, example-rich prompts.
- It is designed to be both robust against crashes and highly accurate in its extraction.
- This is the final, production-ready script.
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

# ---------- 2. SPECIALISED AGENT FUNCTIONS (WITH HIGH-QUALITY PROMPTS) ----------

def agent_extract_metadata(full_text: str) -> dict | None:
    """Agent 1 – extract high-level study metadata."""
    prompt = f"""From the beginning of this document, extract the study information (author, year, journal, design, population, condition, intervention, comparison). If a value is absent, use null. Respond with a JSON object with a key "study_info".

Text to analyse:
{full_text[:8000]}"""
    return parse_json_response(ask_llm(prompt), "study_info")

def agent_locate_defined_outcomes(full_text: str) -> list:
    """Agent 2 – locate planned outcomes described in the Methods section."""
    prompt = f"""From the 'Methods' section of the document, extract all defined outcomes.
RULES:
1. Handle semicolon-separated lists as separate domains (e.g., 'A; B; and C' are three domains).
2. Handle time-based groupings as separate domains (e.g., '...before 34 weeks' and '...before 37 weeks' are two distinct domains).
Respond with a JSON object containing a list called "defined_outcomes".

Document text to analyse:
{full_text}"""
    return parse_json_response(ask_llm(prompt), "defined_outcomes") or []

def agent_parse_table(table_text: str) -> list:
    """Agent 3 – parse a single table and extract outcome names with high fidelity."""
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
Adverse outcomes at <37 wk of gestation
Any — no. (%) 79 (9.9)
'''
- REQUIRED JSON OUTPUT:
`{{
  "table_outcomes": [
    {{"outcome_domain": "Adverse outcomes at <34 wk of gestation", "outcome_specific": "Any"}},
    {{"outcome_domain": "Adverse outcomes at <34 wk of gestation", "outcome_specific": "Preeclampsia"}},
    {{"outcome_domain": "Adverse outcomes at <34 wk of gestation", "outcome_specific": "Small-for-gestational-age status without preeclampsia"}},
    {{"outcome_domain": "Adverse outcomes at <37 wk of gestation", "outcome_specific": "Any"}}
  ]
}}`

Respond with a JSON object with a list called "table_outcomes".

TABLE TEXT TO PARSE:
{table_text}"""
    return parse_json_response(ask_llm(prompt), "table_outcomes") or []

def agent_finalize_and_structure(messy_list: list) -> list:
    """Agent 4 – clean, dedupe and structure the combined outcome list."""
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
    """Orchestrate the calls to the four specialist agents."""
    
    study_info = agent_extract_metadata(full_text)
    defined_outcomes = agent_locate_defined_outcomes(full_text)
    
    table_texts = re.findall(r"(Table \d+\..*?)(?=\nTable \d+\.|\Z)", full_text, re.