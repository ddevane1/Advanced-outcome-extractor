#!/usr/bin/env python3
# -------- advanced_extractor.py (v8.0 - final stable version) --------

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
# These functions are now called within the main cached pipeline

def agent_extract_metadata(full_text: str) -> dict:
    prompt = 'You are a metadata extraction specialist...' # Simplified for brevity
    return parse_json_response(ask_llm(prompt), "study_info")

def agent_locate_defined_outcomes(full_text: str) -> list:
    prompt = 'You are a clinical trial protocol analyst...'
    return parse_json_response(ask_llm(prompt), "defined_outcomes") or []

def agent_parse_table(table_text: str) -> list:
    prompt = (
        "You are an expert at parsing clinical trial tables...\n" # Using the same robust prompt from v5.1
        "**STEP 1: CLASSIFY THE TABLE** (Baseline vs. Outcome)\n"
        "**STEP 2: EXTRACT BASED ON CLASSIFICATION**\n"
        "- If BASELINE, return `{\"table_outcomes\": []}`\n"
        "- If OUTCOME, extract clean names, domains, and specific outcomes.\n"
        f"**TABLE TEXT TO PARSE:**\n{table_text}"
    )
    return parse_json_response(ask_llm(prompt), "table_outcomes") or []

def agent_finalize_and_structure(messy_list: list) -> list:
    prompt = (
        "You are a data structuring expert. Clean, deduplicate, and structure this messy list of outcomes into a final hierarchical list.\n"
        "**OUTPUT FORMAT:** Return a JSON object with key 'final_outcomes'.\n"
        f"**MESSY LIST TO PROCESS:**\n{json.dumps(messy_list)}"
    )
    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "final_outcomes") or []


# ---------- 3. MAIN ORCHESTRATION PIPELINE (CACHED) ----------

@st.cache_data(show_spinner="Running AI extraction pipeline...")
def run_extraction_pipeline(full_text: str):
    """Or