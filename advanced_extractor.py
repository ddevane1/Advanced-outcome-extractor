#!/usr/bin/env python3
# -------- v9.1 - Step 2: Adding the first AI agent --------

import streamlit as st
import pdfplumber
import io
import json
from openai import OpenAI
import os

# --- OpenAI Client Initialization ---
# This line is crucial for the AI to work.
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))


# --- Helper Functions ---

def pdf_to_text(file_bytes):
    """Extracts text from the bytes of an uploaded PDF file."""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
            if not text.strip():
                st.error("This PDF appears to be a scanned image or contains no text.")
                return None
            return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def ask_llm(prompt: str) -> str:
    """Generic function to call the OpenAI API."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4000,
            response_format={"type": "json_object"}
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


# --- The First AI Agent ---

def agent_extract_metadata(full_text: str) -> dict:
    """This agent extracts the high-level study metadata."""
    prompt = (
        "You are a metadata extraction specialist. From the beginning of this document, extract the study information. If a value is absent, use null.\n"
        'Respond in this exact JSON format: {"study_info": {"first_author_surname": "...", "publication_year": "...", "journal": "...", "study_design": "...", "study_country": "...", "patient_population": "...", "targeted_condition": "...", "diagnostic_criteria": "...", "interventions_tested": "...", "comparison_group": "..."}}\n\n'
        f"Text to analyze:\n{full_text[:6000]}"
    )
    
    st.info("Calling the first AI agent to extract metadata...")
    response = ask_llm(prompt)
    return parse_json_response(response, "study_info")


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Debugging Step 2: Testing the Metadata Agent")
st.markdown("This test will see if the first AI agent can run without causing a syntax error.")

uploaded_file = st.file_uploader("Upload a PDF file to test", type="pdf")

if uploaded_file is not None:
    file_contents = uploaded_file.getvalue()
    full_text = pdf_to_text(file_contents)
    
    if full_text:
        # We now call the first agent and display its output.
        metadata = agent_extract_metadata(full_text)
        
        if metadata:
            st.success("Success! The first AI agent ran without errors.")
            st.markdown("The extracted metadata is:")
            st.json(metadata)
            st.info("We can now proceed to add the next agent.")
        else:
            st.error("The agent ran but failed to extract any metadata.")