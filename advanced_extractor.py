#!/usr/bin/env python3
# -------- v9.0 - A simple skeleton to test the foundation --------

import streamlit as st
import pdfplumber
import io

# This function reads the PDF. We are testing this part first.
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

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Debugging Step 1: Basic App Skeleton")
st.markdown("The only purpose of this test is to see if a simple app can run without syntax errors.")

uploaded_file = st.file_uploader("Upload a PDF file to test", type="pdf")

if uploaded_file is not None:
    # We will not process the file yet.
    # We are just checking if the app runs and displays this message.
    st.success("File uploaded successfully!")
    st.success("The basic skeleton app is working without any syntax errors.")
    st.info("Now we will add the AI agents back one by one in the next step.")