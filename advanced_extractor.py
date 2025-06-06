#!/usr/bin/env python3
# -------- advanced_extractor.py (v11.1 - final table parsing fix) --------

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
        st.warning(f"Could not parse a valid JSON response from the AI.")
        return None


# ---------- 2. SPECIALIZED AGENT FUNCTIONS ----------

def agent_extract_metadata(full_text: str) -> dict:
    """Agent 1: Extracts the high-level study metadata."""
    prompt = ('You are a metadata extraction specialist...')
    return parse_json_response(ask_llm(prompt), "study_info")

def agent_locate_defined_outcomes(full_text: str) -> list:
    """Agent 2: Finds planned outcomes from the Methods section."""
    prompt = ('You are a clinical trial protocol analyst...')
    return parse_json_response(ask_llm(prompt), "defined_outcomes") or []

def agent_parse_table(table_text: str) -> list:
    """Agent 3: A specialist agent to parse a single table, with enhanced hierarchy detection."""
    prompt = (
        "You are an expert at parsing clinical trial tables. Analyze the single table text below.\n\n"
        "**STEP 1: CLASSIFY THE TABLE**\n"
        "First, determine if this table describes **baseline patient characteristics** or **clinical trial outcomes**.\n"
        "If it is a BASELINE table, you MUST return an empty list: `{\"table_outcomes\": []}`\n\n"
        "**STEP 2: EXTRACT OUTCOMES (ONLY if it is an outcome table)**\n"
        "**CRITICAL RULES:**\n"
        "1.  **Extract Clean Names:** The outcome name is the text description ONLY. You **MUST STRIP AWAY** all trailing data like '— no. (%)'.\n"
        "2.  **Understand Hierarchy:** A bolded heading is a 'domain'. Items listed under that heading are its 'specific outcomes'. An item can be both a domain and a specific outcome if it has further sub-points.\n\n"
        "--- **DETAILED EXAMPLE OF HIERARCHY** ---\n"
        "**IF THE INPUT TABLE TEXT IS:**\n"
        "'''\nAdverse outcomes at <34 wk of gestation\n"
        "Any — no. (%) 32 (4.0) 53 (6.4)\n"
        "Preeclampsia — no. (%) 3 (0.4) 15 (1.8)\n"
        "Gestational hypertension — no. (%) 2 (0.3) 2 (0.2)\n'''\n\n"
        "**YOUR JSON OUTPUT MUST BE:**\n"
        "```json\n"
        "{\n"
        '  "table_outcomes": [\n'
        '    {"outcome_domain": "Adverse outcomes at <34 wk of gestation", "outcome_specific": "Any"},\n'
        '    {"outcome_domain": "Adverse outcomes at <34 wk of gestation", "outcome_specific": "Preeclampsia"},\n'
        '    {"outcome_domain": "Adverse outcomes at <34 wk of gestation", "outcome_specific": "Gestational hypertension"}\n'
        "  ]\n"
        "}\n"
        "```\n"
        "--- END OF EXAMPLE ---\n\n"
        "**OUTPUT FORMAT:** Return a JSON object with a list called 'table_outcomes'. This list will be empty for baseline tables.\n\n"
        f"**TABLE TEXT TO PARSE:**\n{table_text}"
    )
    return parse_json_response(ask_llm(prompt), "table_outcomes") or []

def agent_finalize_and_structure(messy_list: list) -> list:
    """Agent 4: Takes a messy list of outcomes and cleans, deduplicates, and structures it."""
    prompt = ('You are a data structuring expert. Clean, deduplicate, and structure this messy list of outcomes into a final hierarchical list...\n'
              f'**MESSY LIST TO PROCESS:**\n{json.dumps(messy_list, indent=2)}')
    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "final_outcomes") or []


# ---------- 3. MAIN ORCHESTRATION PIPELINE (CACHED) ----------

@st.cache_data(show_spinner="Step 2: Running AI extraction pipeline...")
def run_extraction_pipeline(full_text: str):
    """Orchestrates the AI agent calls. This entire function is cached."""
    
    study_info = agent_extract_metadata(full_text)
    defined_outcomes = agent_locate_defined_outcomes(full_text)
    
    table_texts = re.findall(r"(Table \d+\..*?)(?=\nTable \d+\.|\Z)", full_text, re.DOTALL)
    all_table_outcomes = []
    if table_texts:
        for table_text in table_texts:
            parsed_outcomes = agent_parse_table(table_text)
            if parsed_outcomes:
                all_table_outcomes.extend(parsed_outcomes)
    
    raw_combined_list = defined_outcomes + all_table_outcomes
    if not raw_combined_list:
        return study_info, []

    final_outcomes = agent_finalize_and_structure(raw_combined_list)
    return study_info, final_outcomes


# ---------- 4. STREAMLIT UI ----------

st.set_page_config(layout="wide")
st.title("Clinical Trial Outcome Extractor (v11.1)")
st.markdown("This tool uses a cached, multi-agent AI workflow to accurately and reliably extract outcomes.")

uploaded_file = st.file_uploader("Upload a PDF clinical trial report to begin", type="pdf")

if uploaded_file is not None:
    file_contents = uploaded_file.getvalue()
    full_text = get_pdf_text(file_contents)
    
    if full_text:
        study_info, outcomes = run_extraction_pipeline(full_text)

        if outcomes:
            st.success(f"Processing complete for **{uploaded_file.name}**.")
            df = pd.DataFrame(outcomes)
            for col in ['outcome_domain', 'outcome_specific', 'outcome_type', 'definition', 'timepoint']:
                if col not in df.columns: df[col] = ''
            df.fillna('', inplace=True)

            # HIERARCHICAL DISPLAY IN-APP
            st.subheader("Hierarchical Outcome View")
            domains = df[df['outcome_domain'] != '']['outcome_domain'].unique()
            for domain in domains:
                st.markdown(f"**DOMAIN:** {domain}")
                specific_outcomes = df[(df['outcome_domain'] == domain) & (df['outcome_specific'] != '') & (df['outcome_specific'] != domain)]['outcome_specific'].unique()
                if len(specific_outcomes) > 0:
                    for specific in specific_outcomes:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {specific}")
                else:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• *This is a primary outcome or a domain with no specific sub-outcomes listed.*")
                st.write("") 

            # PUBLICATION-READY EXPORT
            st.subheader("Export Results")
            export_rows = []
            for domain in domains:
                domain_row = df[df['outcome_domain'] == domain].iloc[0]
                export_rows.append({"Domain": domain, "Specific Outcome": "", "Definition": domain_row.get('definition', ''), "Timepoint": domain_row.get('timepoint', '')})
                specific_outcomes_df = df[(df['outcome_domain'] == domain) & (df['outcome_specific'] != '') & (df['outcome_specific'] != domain)]
                for _, specific_row in specific_outcomes_df.iterrows():
                    export_rows.append({"Domain": "", "Specific Outcome": specific_row['outcome_specific'], "Definition": specific_row.get('definition', ''), "Timepoint": specific_row.get('timepoint', '')})
            export_df = pd.DataFrame(export_rows)

            st.download_button(
                label="**Download Publication-Ready CSV**",
                data=export_df.to_csv(index=False).encode('utf-8'),
                file_name=f"Publication_Outcomes_{uploaded_file.name}.csv",
                mime='text/csv',
                help="A clean, human-readable table with domains listed once, followed by their specific outcomes."
            )

            # EXPANDER FOR METADATA AND RAW DATA
            with st.expander("Show Extracted Study Information"):
                st.json(study_info or {})
            with st.expander("Show Full Raw Data Table (for analysis)"):
                st.dataframe(df)
        else:
            st.error("Extraction ran but no outcomes were found.")