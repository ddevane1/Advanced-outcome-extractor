#!/usr/bin/env python3
# -------- advanced_extractor.py (v14.0 - Final attempt with text parsing) --------

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

def ask_llm(prompt: str, max_response_tokens: int = DEFAULT_TOKENS_FOR_RESPONSE) -> str:
    """Generic function to call the OpenAI API, requesting plain text."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_response_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An API error occurred: {e}")
        return None

def parse_json_response(response_text: str, key: str):
    """Safely finds and parses a JSON block from a larger text response."""
    if not response_text: return None
    
    # Use regex to find the json block wrapped in markdown
    match = re.search(r"```json\n({.*?})\n```", response_text, re.DOTALL)
    
    if not match:
        st.warning("Could not find a valid JSON block in the AI's text response.")
        return None
        
    json_str = match.group(1)
    try:
        data = json.loads(json_str)
        return data if key is None else data.get(key)
    except json.JSONDecodeError:
        st.warning("Failed to decode the extracted JSON block.")
        return None


# ---------- 2. SPECIALIZED AGENT FUNCTIONS ----------

def agent_extract_metadata(full_text: str) -> dict:
    """Agent 1: Extracts the high-level study metadata."""
    prompt = f"""You are a metadata extraction specialist. From the beginning of this document, extract the study information. If a value is absent, use null.
You must wrap your final JSON object in markdown like this:
```json
{{
  "study_info": {{...}}
}}