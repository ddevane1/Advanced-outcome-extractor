#!/usr/bin/env python3

"""
Universal Clinical‑Trial Extractor – **v12.0.1 hot‑fix**
──────────────────────────────────────────────────────
*This is your original v12.0 code, unchanged except for two critical bug‑fixes*:

1. **SyntaxError at `parse_json_response` (line 58)**  
   The line now reads:  
   ```python
   json_str = response_text.strip().removeprefix("```json").removesuffix("```")
   ```
   so the string literal is properly terminated.

2. **OpenAI “json must appear” requirement**  
   `ask_llm()` now prepends a trivial system message containing the word
   “json”, ensuring the 400‑error does not occur. No other prompt content was
   touched.

Everything else — agents, regexes, UI — is *exactly* as in your pasted v12.0.
"""

import os
import json
import re
import io

import pdfplumber
import pandas as pd
import streamlit as st
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
    st.info("Step 1 • Reading PDF text…")
    try:
        with pdfplumber.open(io.BytesIO(file_contents)) as pdf:
            full_text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
            if not full_text.strip():
                st.error("This PDF appears to be a scanned image or contains no extractable text.")
                return None
            st.success("✓ PDF text read successfully.")
            return full_text
    except Exception as e:  # noqa: BLE001
        st.error(f"PDF read error: {e}")
        return None


def ask_llm(prompt: str, *, is_json: bool = True, max_response_tokens: int = DEFAULT_TOKENS_FOR_RESPONSE):
    """Call OpenAI with the mandatory ‘json’ system message to avoid HTTP 400."""
    messages = [
        {"role": "system", "content": "You are a json‑only assistant; always respond with valid JSON."},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=max_response_tokens,
            response_format={"type": "json_object"} if is_json else {"type": "text"},
        )
        return resp.choices[0].message.content
    except Exception as e:  # noqa: BLE001
        st.error(f"OpenAI error: {e}")
        return None


def parse_json_response(response_text: str, key: str | None = None):
    """Safely parse JSON from the LLM response."""
    if not response_text:
        return None
    try:
        json_str = response_text.strip().removeprefix("```json").removesuffix("```")
        data = json.loads(json_str)
        return data if key is None else data.get(key)
    except (json.JSONDecodeError, AttributeError):  # noqa: PERF203
        st.warning("Could not parse a valid JSON response from the AI.")
        return None

# ---------- 2. UNIVERSAL SPECIALISED AGENT FUNCTIONS ----------
# … **(identical to your original v12.0 agents, omitted here for brevity)** …

# ---------- 3. UNIVERSAL TABLE DETECTION ----------
# … unchanged …

# ---------- 4. MAIN ORCHESTRATION PIPELINE ----------
# … unchanged …

# ---------- 5. STREAMLIT UI ----------
# … unchanged …
