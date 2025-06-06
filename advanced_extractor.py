#!/usr/bin/env python3
# -------- enhanced_study_extractor.py (v12.0 - comprehensive study metadata + outcomes) --------

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

def parse_json_response(response_text: str, key: str = None):
    """Safely parses JSON from the LLM response."""
    if not response_text: return None
    try:
        json_str = response_text.strip().removeprefix("```json").removesuffix("```")
        data = json.loads(json_str)
        return data if key is None else data.get(key)
    except (json.JSONDecodeError, AttributeError):
        st.warning(f"Could not parse a valid JSON response from the AI.")
        return None

# ---------- 2. ENHANCED SPECIALIZED AGENT FUNCTIONS ----------

def agent_extract_comprehensive_study_metadata(full_text: str) -> dict:
    """Agent 1: Extracts comprehensive study-level metadata including all requested fields."""
    prompt = f'''You are a comprehensive study metadata extraction specialist. From this clinical trial document, extract ALL the study-level information listed below. If any value is not found or unclear, use null.

**REQUIRED STUDY-LEVEL FIELDS:**
- Last author + year (e.g., "Smith 2023")
- Paper title (complete title)
- Journal (journal name)
- Healthcare setting (hospital, community, primary care, etc.)
- Country participants recruited in
- Patient population (description of study participants)
- Targeted condition with definition
- Intervention tested (what was being tested)
- Comparator (control group/comparison intervention)

**SEARCH STRATEGY:**
- Look in title page, abstract, methods section, author affiliations
- For last author: Find the final author in the author list
- For healthcare setting: Look for hospital names, community settings, primary care mentions
- For country: Check author affiliations, study sites, methods section
- For targeted condition: Look for condition definitions in background/methods
- For interventions: Check methods, abstract, study design sections

Respond in this exact JSON format:
{{
  "study_metadata": {{
    "last_author_year": "...",
    "paper_title": "...",
    "journal": "...",
    "healthcare_setting": "...",
    "country": "...",
    "patient_population": "...",
    "targeted_condition": "...",
    "intervention_tested": "...",
    "comparator": "..."
  }}
}}

**Document Text to Analyze (first 12,000 characters):**
{full_text[:12000]}'''

    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "study_metadata")

def agent_locate_defined_outcomes(full_text: str) -> list:
    """Agent 2: Finds planned outcomes from the Methods section."""
    prompt = ('You are a clinical trial protocol analyst. Extract all outcome definitions, typically found in the \'Methods\' section.\n\n'
              '**RULES:**\n'
              "1.  **Handle Semicolon-Separated Lists:** Treat each item in a semicolon-separated list as a separate outcome domain.\n"
              "2.  **Handle Time-Based Grouping:** Create a separate domain for each timepoint (e.g., 'before 34 weeks').\n\n"
              '**OUTPUT FORMAT:** Return a JSON object with a list called \'defined_outcomes\'.\n\n'
              f'**Document Text to Analyze:**\n{full_text}')
    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "defined_outcomes") or []

def agent_parse_table_enhanced(table_text: str) -> list:
    """Agent 3: Enhanced table parser that captures hierarchical structure and complete names."""
    prompt = f'''You are an expert at parsing clinical trial tables with PERFECT hierarchical structure recognition.

**STEP 1: CLASSIFY THE TABLE**
First, determine if this table describes **baseline patient characteristics** (demographics, age, etc.) or **clinical trial outcomes** (results, events, complications, adverse events).

**STEP 2: EXTRACT TRUE HIERARCHICAL STRUCTURE**

**If BASELINE table:** Return `{{"table_outcomes": []}}`

**If OUTCOME table:** Extract with these PRECISE rules:

1. **IDENTIFY TRUE HIERARCHY FROM TABLE LAYOUT:**
   Look for this pattern:
   ```
   Main Header (often bold/caps)               <- This is outcome_domain
       Sub-item 1 — no. (%)                   <- This is outcome_specific  
       Sub-item 2 — no. (%)                   <- This is outcome_specific
       Sub-item 3 — no. (%)                   <- This is outcome_specific
   Another Main Header                         <- This is outcome_domain
       Different sub-item — no. (%)           <- This is outcome_specific
   ```

2. **PRESERVE COMPLETE NAMES INCLUDING ALL QUALIFIERS:**
   - "Miscarriage or stillbirth without preeclampsia — no. (%)" becomes:
     * outcome_specific: "Miscarriage or stillbirth without preeclampsia"
   - "Small-for-gestational-age status without preeclampsia — no./total no. (%)" becomes:
     * outcome_specific: "Small-for-gestational-age status without preeclampsia"

3. **EXAMPLES OF CORRECT EXTRACTION:**
   From table text like:
   ```
   Adverse outcomes at <37 wk of gestation
       Any — no. (%)                          79 (9.9)    116 (14.1)
       Gestational hypertension — no. (%)     8 (1.0)     7 (0.9)
       Small-for-gestational-age status without preeclampsia — no./total no. (%)  17/785 (2.2)  18/807 (2.2)
       Miscarriage or stillbirth without preeclampsia — no. (%)  14 (1.8)  19 (2.3)
   ```
   
   Should extract:
   - domain="Adverse outcomes at <37 wk of gestation", specific="Any"
   - domain="Adverse outcomes at <37 wk of gestation", specific="Gestational hypertension"  
   - domain="Adverse outcomes at <37 wk of gestation", specific="Small-for-gestational-age status without preeclampsia"
   - domain="Adverse outcomes at <37 wk of gestation", specific="Miscarriage or stillbirth without preeclampsia"

4. **HANDLE NESTED STRUCTURE:**
   ```
   Stillbirth or death — no. (%)
       All stillbirths or deaths              8 (1.0)     14 (1.7)
       With preeclampsia or status of being small for gestational age  5 (0.6)  8 (1.0)
       Without preeclampsia or status of being small for gestational age  3 (0.4)  6 (0.7)
   ```
   
   Should extract:
   - domain="Stillbirth or death", specific="All stillbirths or deaths"
   - domain="Stillbirth or death", specific="With preeclampsia or status of being small for gestational age"
   - domain="Stillbirth or death", specific="Without preeclampsia or status of being small for gestational age"

**OUTPUT FORMAT:** Return a JSON object with a list called 'table_outcomes':
{{
  "outcome_domain": "Exact main section header",
  "outcome_specific": "Exact sub-item text (complete with all qualifiers)",
  "definition": "...",
  "measurement_method": "...",
  "timepoint": "..."
}}

**TABLE TEXT TO PARSE:**
{table_text}'''

    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "table_outcomes") or []

def agent_finalize_and_structure_enhanced(messy_list: list) -> list:
    """Agent 4: Enhanced structuring that preserves hierarchical structure and complete names."""
    prompt = f'''You are a data structuring expert. Clean, deduplicate, and structure this messy list of outcomes while PRESERVING hierarchical structure and complete outcome names.

**CRITICAL PRESERVATION RULES:**
1. **PRESERVE EXACT HIERARCHICAL STRUCTURE:**
   - Keep domain-specific relationships EXACTLY as extracted
   - "Adverse outcomes at <37 wk of gestation" should remain as domain with its specific sub-outcomes
   - "Stillbirth or death" should remain as domain with its specific sub-outcomes
   - DO NOT create generic domains

2. **PRESERVE COMPLETE OUTCOME NAMES:**
   - "Miscarriage or stillbirth without preeclampsia" must stay EXACTLY as is
   - "Small-for-gestational-age status without preeclampsia" must stay EXACTLY as is
   - DO NOT truncate or simplify these names

3. **OUTCOME STRUCTURE:**
   - For each unique outcome domain, create one entry with `"outcome_type": "domain"`
   - For each specific outcome under a domain, create an entry with `"outcome_type": "specific"`
   - Combine information if you see the same outcome multiple times

**OUTPUT FORMAT:** Return a final JSON object with a key 'final_outcomes'. Each item must have keys: 'outcome_type', 'outcome_domain', 'outcome_specific', 'definition', 'measurement_method', 'timepoint'.

**MESSY LIST TO PROCESS:**
{json.dumps(messy_list, indent=2)}'''

    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "final_outcomes") or []

# ---------- 3. MAIN ORCHESTRATION PIPELINE (CACHED) ----------

@st.cache_data(show_spinner="Step 2: Running comprehensive AI extraction pipeline...")
def run_comprehensive_extraction_pipeline(full_text: str):
    """Orchestrates the AI agent calls for comprehensive extraction."""
    
    # Extract comprehensive study metadata
    study_metadata = agent_extract_comprehensive_study_metadata(full_text)
    
    # Extract defined outcomes
    defined_outcomes = agent_locate_defined_outcomes(full_text)
    
    # Extract and parse all tables
    table_texts = re.findall(r"(Table \d+\..*?)(?=\nTable \d+\.|\Z)", full_text, re.DOTALL)
    all_table_outcomes = []
    if table_texts:
        for table_text in table_texts:
            parsed_outcomes = agent_parse_table_enhanced(table_text)
            if parsed_outcomes:
                all_table_outcomes.extend(parsed_outcomes)
    
    # Combine all outcomes
    raw_combined_list = defined_outcomes + all_table_outcomes
    if not raw_combined_list:
        return study_metadata, []

    # Final structuring
    final_outcomes = agent_finalize_and_structure_enhanced(raw_combined_list)
    return study_metadata, final_outcomes

# ---------- 4. ENHANCED STREAMLIT UI ----------

st.set_page_config(layout="wide")
st.title("Enhanced Clinical Trial Data Extractor (v12.0)")
st.markdown("""
**✨ ENHANCED FEATURES:**
- **Comprehensive Study Metadata:** Extracts last author+year, title, journal, healthcare setting, country, etc.
- **Hierarchical Outcomes:** Preserves exact domain-specific structure (e.g., adverse events → specific events)
- **Complete Names:** Maintains full outcome descriptions including qualifiers
- **Publication Ready:** Clean export format for systematic reviews
""")

uploaded_file = st.file_uploader("Upload a PDF clinical trial report to begin", type="pdf")

if uploaded_file is not None:
    file_contents = uploaded_file.getvalue()
    full_text = get_pdf_text(file_contents)
    
    if full_text:
        study_metadata, outcomes = run_comprehensive_extraction_pipeline(full_text)

        if outcomes:
            st.success(f"✅ Processing complete for **{uploaded_file.name}**")
            
            # Display comprehensive study metadata
            st.subheader("📋 Study Information")
            if study_metadata:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Last Author + Year:** {study_metadata.get('last_author_year', 'Not found')}")
                    st.write(f"**Paper Title:** {study_metadata.get('paper_title', 'Not found')}")
                    st.write(f"**Journal:** {study_metadata.get('journal', 'Not found')}")
                    st.write(f"**Healthcare Setting:** {study_metadata.get('healthcare_setting', 'Not found')}")
                    st.write(f"**Country:** {study_metadata.get('country', 'Not found')}")
                
                with col2:
                    st.write(f"**Patient Population:** {study_metadata.get('patient_population', 'Not found')}")
                    st.write(f"**Targeted Condition:** {study_metadata.get('targeted_condition', 'Not found')}")
                    st.write(f"**Intervention:** {study_metadata.get('intervention_tested', 'Not found')}")
                    st.write(f"**Comparator:** {study_metadata.get('comparator', 'Not found')}")
            
            # Display outcomes with hierarchical structure
            st.subheader("🎯 Extracted Outcomes")
            df = pd.DataFrame(outcomes)
            for col in ['outcome_domain', 'outcome_specific', 'outcome_type', 'definition', 'timepoint']:
                if col not in df.columns: 
                    df[col] = ''
            df.fillna('', inplace=True)

            # HIERARCHICAL DISPLAY IN-APP
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

            # ENHANCED EXPORT with study metadata at beginning
            st.subheader("📊 Export Data")
            
            # Create export structure: Study metadata at top, then outcomes
            export_rows = []
            
            # Add study metadata row first
            if study_metadata:
                study_row = {
                    "Data_Type": "STUDY_METADATA",
                    "Last_Author_Year": study_metadata.get('last_author_year', ''),
                    "Paper_Title": study_metadata.get('paper_title', ''),
                    "Journal": study_metadata.get('journal', ''),
                    "Healthcare_Setting": study_metadata.get('healthcare_setting', ''),
                    "Country": study_metadata.get('country', ''),
                    "Patient_Population": study_metadata.get('patient_population', ''),
                    "Targeted_Condition": study_metadata.get('targeted_condition', ''),
                    "Intervention_Tested": study_metadata.get('intervention_tested', ''),
                    "Comparator": study_metadata.get('comparator', ''),
                    "Outcome_Domain": "",
                    "Outcome_Specific": "",
                    "Definition": "",
                    "Timepoint": ""
                }
                export_rows.append(study_row)
            
            # Add outcomes
            for domain in domains:
                # Find the primary entry for the domain
                domain_row = df[df['outcome_domain'] == domain].iloc[0]
                outcome_row = {
                    "Data_Type": "OUTCOME_DOMAIN",
                    "Last_Author_Year": "",
                    "Paper_Title": "",
                    "Journal": "",
                    "Healthcare_Setting": "",
                    "Country": "",
                    "Patient_Population": "",
                    "Targeted_Condition": "",
                    "Intervention_Tested": "",
                    "Comparator": "",
                    "Outcome_Domain": domain,
                    "Outcome_Specific": "",
                    "Definition": domain_row.get('definition', ''),
                    "Timepoint": domain_row.get('timepoint', '')
                }
                export_rows.append(outcome_row)
                
                # Add specific outcomes for this domain
                specific_outcomes_df = df[(df['outcome_domain'] == domain) & (df['outcome_specific'] != '') & (df['outcome_specific'] != domain)]
                for _, specific_row in specific_outcomes_df.iterrows():
                    specific_outcome_row = {
                        "Data_Type": "OUTCOME_SPECIFIC",
                        "Last_Author_Year": "",
                        "Paper_Title": "",
                        "Journal": "",
                        "Healthcare_Setting": "",
                        "Country": "",
                        "Patient_Population": "",
                        "Targeted_Condition": "",
                        "Intervention_Tested": "",
                        "Comparator": "",
                        "Outcome_Domain": "",
                        "Outcome_Specific": specific_row['outcome_specific'],
                        "Definition": specific_row.get('definition', ''),
                        "Timepoint": specific_row.get('timepoint', '')
                    }
                    export_rows.append(specific_outcome_row)
            
            export_df = pd.DataFrame(export_rows)

            st.download_button(
                label="📥 **Download Complete Dataset (CSV)**",
                data=export_df.to_csv(index=False).encode('utf-8'),
                file_name=f"Complete_Study_Data_{uploaded_file.name.replace('.pdf', '')}.csv",
                mime='text/csv',
                help="Complete dataset: Study metadata at top, then hierarchical outcomes"
            )
            
            # Show preview
            with st.expander("👁️ Preview Export Data"):
                st.dataframe(export_df.head(10))
                st.info(f"Total rows: {len(export_df)} (Study metadata + outcome domains + specific outcomes)")

        else:
            st.warning("⚠️ No outcomes were extracted. The document may not contain recognizable outcome data.")
            
            # Still show metadata if available
            if study_metadata:
                st.subheader("📋 Study Metadata Found")
                for key, value in study_metadata.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")