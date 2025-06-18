#!/usr/bin/env python3

# -------- enhanced_clinical_trial_extractor_v15.py (with Marker) --------

import os
import json
import re
import streamlit as st
import pandas as pd
import io
import tempfile
import subprocess
import sys
from pathlib import Path
from openai import OpenAI

# ----- CONFIG -----
MODEL = "o"
DEFAULT_TOKENS_FOR_RESPONSE = 4096
LARGE_TOKENS_FOR_RESPONSE = 8192
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))

# ---------- 1. ENHANCED PDF PROCESSING WITH MARKER ----------

@st.cache_data
def convert_pdf_to_markdown_with_marker(file_contents: bytes) -> str:
    """
    Converts PDF to markdown using Marker for better structure preservation.
    Falls back to pdfplumber if Marker fails.
    """
    st.info("Step 1: Converting PDF to structured markdown...")
    
    try:
        # Create temporary file for PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_pdf:
            tmp_pdf.write(file_contents)
            tmp_pdf_path = tmp_pdf.name
        
        # Create output directory
        output_dir = tempfile.mkdtemp()
        
        # Run Marker conversion with LLM enhancement
        st.info("🔄 Running Marker conversion with LLM enhancement (this may take 60-120 seconds for highest accuracy)...")
        
        try:
            # Install marker if not available
            try:
                import marker
            except ImportError:
                st.info("Installing Marker dependency...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "marker-pdf"])
                import marker
            
            # Run marker conversion with LLM enhancement
            from marker.convert import convert_single_pdf
            from marker.models import load_all_models
            
            # Load models (cached after first use) - now includes LLM models
            model_lst = load_all_models()
            
            # Convert PDF with LLM enhancement for highest accuracy
            full_text, images, out_meta = convert_single_pdf(
                tmp_pdf_path, 
                model_lst, 
                max_pages=None,
                langs=None,
                batch_multiplier=1,
                start_page=None,
                use_llm=True  # Enable LLM for highest accuracy
            )
            
            # Clean up temp file
            os.unlink(tmp_pdf_path)
            
            if not full_text.strip():
                raise Exception("Marker returned empty text")
                
            st.success("✓ PDF converted to structured markdown with Marker + LLM enhancement")
            return full_text
            
        except Exception as marker_error:
            st.warning(f"Marker conversion failed: {marker_error}")
            st.info("Falling back to pdfplumber...")
            return fallback_to_pdfplumber(file_contents)
            
    except Exception as e:
        st.error(f"PDF processing failed: {e}")
        return fallback_to_pdfplumber(file_contents)

def fallback_to_pdfplumber(file_contents: bytes) -> str:
    """Fallback PDF text extraction using pdfplumber."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_contents)) as pdf:
            full_text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
            if not full_text.strip():
                st.error("This PDF appears to be a scanned image or contains no extractable text.")
                return None
            st.success("✓ PDF text extracted with pdfplumber (fallback)")
            return full_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# ---------- 2. ENHANCED AGENT FUNCTIONS FOR MARKDOWN ----------

def agent_extract_comprehensive_study_metadata_md(markdown_text: str) -> dict:
    """Agent 1: Enhanced metadata extraction leveraging markdown structure."""
    prompt = f'''You are a comprehensive study metadata extraction specialist. This document has been converted to structured markdown, making it easier to identify sections and extract information.

**MARKDOWN ADVANTAGES:**
- Headers are clearly marked with # ## ### 
- Tables are properly formatted
- Lists and structure are preserved
- Author information is cleanly formatted

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

**ENHANCED SEARCH STRATEGY FOR MARKDOWN:**
- Title: Look for # headers at document start
- Authors: Look for author lists near title, last author is final in list
- Journal: Look for journal information in header/footer
- Healthcare setting: Search for hospital names, "community-based", "primary care"
- Country: Check Methods sections, author affiliations
- Study details: Use markdown structure to navigate to Methods, Background sections

**SECTION IDENTIFICATION:**
Use markdown headers to locate:
- # Title or main header
- ## Abstract, ## Introduction, ## Methods
- ## Results, ## Discussion
- Author affiliations and institutional information

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

**Structured Markdown Document to Analyze:**
{markdown_text[:15000]}'''

    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "study_metadata")

def agent_locate_defined_outcomes_with_markdown_structure(markdown_text: str) -> list:
    """Agent 2: Enhanced outcome extraction using markdown structure."""
    prompt = f'''You are a clinical trial outcome extraction specialist. This structured markdown document makes it much easier to identify outcome definitions by using headers and formatted sections.

**MARKDOWN STRUCTURE ADVANTAGES:**
- ## Methods section clearly marked
- ### Outcome Measures subsections identified
- **Bold text** for primary/secondary outcomes
- Proper list formatting for multiple outcomes
- Table structure preserved for outcome details

**ENHANCED EXTRACTION STRATEGY:**

1. **Navigate by Headers:**
   - Find ## Methods, ## Study Design sections
   - Look for ### Primary Outcome, ### Secondary Outcomes
   - Check ### Endpoints, ### Outcome Measures subsections

2. **Use Formatting Cues:**
   - **Bold text** often indicates outcome names
   - Numbered/bulleted lists for multiple outcomes
   - Italics for definitions or timepoints

3. **Table Structure:**
   - Markdown tables clearly show outcome hierarchies
   - Headers indicate timepoints and categories
   - Rows show specific outcomes with definitions

4. **Follow-up Outcomes:**
   - Navigate to ## Results section using headers
   - Look for mortality, adverse events, censures
   - Check ## Safety, ## Follow-up sections

**WHAT TO EXTRACT:**
- Primary outcome definitions and timepoints
- Secondary outcome definitions and timepoints  
- Safety outcome definitions and timepoints
- Follow-up outcomes: Death, mortality, censures, adverse events
- Measurement instruments and assessment timepoints
- Diagnostic criteria and measurement methods

**OUTPUT FORMAT:** Return JSON with 'defined_outcomes' list:
{{
  "outcome_domain": "Main outcome category",
  "outcome_specific": "Specific measure if applicable",
  "definition": "HOW the outcome is defined/measured",
  "measurement_method": "Instrument/scale/criteria used",
  "timepoint": "WHEN outcome is measured/assessed"
}}

**Structured Markdown Document to Analyze:**
{markdown_text}'''

    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "defined_outcomes") or []

def agent_parse_markdown_tables_enhanced(markdown_text: str) -> list:
    """Agent 3: Enhanced table parsing for markdown-formatted tables."""
    
    # Extract all markdown tables
    table_pattern = r'\|.*?\|\s*\n\|[-:\s|]+\|\s*\n(?:\|.*?\|\s*\n)+'
    tables = re.findall(table_pattern, markdown_text, re.MULTILINE)
    
    if not tables:
        return []
    
    all_table_outcomes = []
    
    for i, table in enumerate(tables):
        prompt = f'''You are an expert at parsing markdown-formatted clinical trial tables. Markdown tables are much cleaner and easier to parse than raw text tables.

**MARKDOWN TABLE ADVANTAGES:**
- Clear column separation with | delimiters
- Header rows clearly defined
- Hierarchical structure preserved
- Better alignment and formatting

**STEP 1: CLASSIFY THE TABLE**
Determine if this table describes **baseline characteristics** or **clinical trial outcomes**.

**STEP 2: EXTRACT WITH PRECISION**

**If BASELINE table:** Return `{{"table_outcomes": []}}`

**If OUTCOME table:** Extract following these rules:

1. **PARSE MARKDOWN STRUCTURE:**
   ```
   | Outcome Category | Group A | Group B |
   |------------------|---------|---------|
   | Primary endpoint | value   | value   |
   | Secondary endpoint| value  | value   |
   ```

2. **IDENTIFY HIERARCHICAL RELATIONSHIPS:**
   - Main categories often span multiple rows
   - Sub-outcomes are indented or listed below main categories
   - Look for outcome groupings and timepoints in headers

3. **EXTRACT COMPLETE INFORMATION:**
   - Preserve full outcome names including qualifiers
   - Capture timepoints from table headers or outcome names
   - Extract measurement details from footnotes or headers

4. **HANDLE SPECIAL FORMATTING:**
   - Bold headers indicate main outcome domains
   - Italics often show definitions or notes
   - Numbers in parentheses usually indicate percentages
   - "no. (%)" format indicates count and percentage data

**OUTPUT FORMAT:** Return JSON with 'table_outcomes' list:
{{
  "outcome_domain": "Main outcome category from table",
  "outcome_specific": "Specific outcome measure",
  "definition": "How outcome is defined (from context)",
  "measurement_method": "Assessment method (if mentioned)",
  "timepoint": "When measured (from headers/context)"
}}

**MARKDOWN TABLE #{i+1} TO PARSE:**
{table}'''

        parsed_outcomes = parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "table_outcomes")
        if parsed_outcomes:
            all_table_outcomes.extend(parsed_outcomes)
    
    return all_table_outcomes

# ---------- 3. ENHANCED ORCHESTRATION PIPELINE ----------

@st.cache_data(show_spinner="Step 2: Running enhanced AI extraction pipeline with structured markdown...")
def run_enhanced_extraction_pipeline(markdown_text: str):
    """Enhanced orchestration using structured markdown."""
    
    # Extract comprehensive study metadata using markdown structure
    study_metadata = agent_extract_comprehensive_study_metadata_md(markdown_text)
    
    # Extract defined outcomes leveraging markdown headers
    defined_outcomes = agent_locate_defined_outcomes_with_markdown_structure(markdown_text)
    
    # Parse markdown tables with enhanced structure awareness
    table_outcomes = agent_parse_markdown_tables_enhanced(markdown_text)
    
    # Extract results outcomes from structured sections
    results_outcomes = agent_extract_results_outcomes_md(markdown_text)
    
    # Combine all outcomes from multiple sources
    raw_combined_list = defined_outcomes + results_outcomes + table_outcomes
    if not raw_combined_list:
        return study_metadata, []

    # Final structuring with enhanced context
    final_outcomes = agent_finalize_comprehensive_structure(raw_combined_list)
    return study_metadata, final_outcomes

def agent_extract_results_outcomes_md(markdown_text: str) -> list:
    """Enhanced results extraction using markdown section navigation."""
    prompt = f'''You are a clinical trial results analyst. Use the structured markdown format to efficiently navigate to Results, Follow-up, and Discussion sections.

**MARKDOWN NAVIGATION STRATEGY:**
- Use ## Results header to find results section
- Look for ### Follow-up, ### Safety subsections
- Navigate to ## Discussion for additional outcomes
- Use markdown structure to identify outcome hierarchies

**WHAT TO EXTRACT:**
Find outcomes that are REPORTED in these sections:
- Death/Mortality rates and causes
- Censures and dropouts
- Adverse events and complications
- Hospital outcomes (length of stay, readmissions)
- Post-hoc analyses and additional endpoints

**ENHANCED SEARCH USING MARKDOWN:**
- Headers clearly delineate sections
- Lists show structured outcome reporting
- Tables are properly formatted for easier parsing
- Bold/italic formatting highlights key outcomes

**OUTPUT FORMAT:** Return JSON with 'results_outcomes' list:
{{
  "outcome_domain": "Main outcome category",
  "outcome_specific": "Specific measure", 
  "definition": "HOW outcome was identified/measured",
  "measurement_method": "Assessment approach",
  "timepoint": "WHEN it occurred/was measured"
}}

**Structured Markdown Document:**
{markdown_text}'''

    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "results_outcomes") or []

# ---------- 4. HELPER FUNCTIONS (UNCHANGED) ----------

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

def agent_finalize_comprehensive_structure(messy_list: list) -> list:
    """Agent 4: Enhanced structuring (same as before)."""
    prompt = f'''You are a data structuring expert. Clean, deduplicate, and structure this messy list of outcomes while PRESERVING hierarchical structure, complete outcome names, definitions, and timepoints.

**CRITICAL PRESERVATION RULES:**

1. **PRESERVE EXACT HIERARCHICAL STRUCTURE:**
   - Keep domain-specific relationships EXACTLY as extracted
   - "Adverse outcomes at <37 wk of gestation" should remain as domain with its specific sub-outcomes
   - DO NOT create generic domains

2. **PRESERVE COMPLETE OUTCOME NAMES:**
   - "Miscarriage or stillbirth without preeclampsia" must stay EXACTLY as is
   - "Respiratory distress syndrome treated with surfactant and ventilation" must stay EXACTLY as is
   - DO NOT truncate or simplify these names

3. **CONSOLIDATE DEFINITIONS AND TIMEPOINTS:**
   - Combine definition information from multiple sources for the same outcome
   - Merge timepoint information intelligently
   - If same outcome appears with different definitions, combine them
   - Standardize similar timepoints: "before 37 weeks", "<37 wk", "prior to 37 weeks" → "before 37 weeks"

4. **OUTCOME STRUCTURE:**
   - For each unique outcome domain, create one entry with "outcome_type": "domain"
   - For each specific outcome under a domain, create an entry with "outcome_type": "specific"
   - Inherit domain timepoints to specific outcomes if they don't have their own

5. **DEFINITION ENHANCEMENT:**
   - Look for outcome definitions from Methods sections
   - Combine table-based information with protocol definitions
   - Preserve diagnostic criteria and measurement details

6. **TIMEPOINT STANDARDIZATION:**
   - "at 30 days", "day 30", "30-day" → "30 days"
   - "before 37 weeks", "<37 wk of gestation", "prior to 37 weeks" → "before 37 weeks of gestation"
   - "at hospital discharge", "discharge", "upon discharge" → "hospital discharge"

**OUTPUT FORMAT:** Return a final JSON object with a key 'final_outcomes'. Each item must have:
{{
  "outcome_type": "domain/specific",
  "outcome_domain": "Exact domain name",
  "outcome_specific": "Exact specific name (or blank for domain-only)",
  "definition": "Complete definition of how outcome is measured/defined",
  "measurement_method": "Instrument/scale/criteria used",
  "timepoint": "When outcome is measured/assessed (standardized)"
}}

**MESSY LIST TO PROCESS:**
{json.dumps(messy_list, indent=2)}'''

    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "final_outcomes") or []

# ---------- 5. ENHANCED STREAMLIT UI ----------

st.set_page_config(layout="wide")
st.title("Enhanced Clinical Trial Data Extractor v14.0 (with Marker + LLM)")
st.markdown("""
**✨ NEW ENHANCED FEATURES:**
- **🔥 Marker + LLM Integration:** Converts PDFs to structured markdown using AI for highest accuracy
- **📋 Comprehensive Study Metadata:** Extracts last author+year, title, journal, healthcare setting, country, etc.
- **🎯 Detailed Outcomes:** Captures definitions, measurement methods, and timepoints for all outcomes
- **🏗️ Hierarchical Structure:** Preserves exact domain-specific structure using markdown formatting
- **📝 Complete Names:** Maintains full outcome descriptions including qualifiers
- **📊 Publication Ready:** Clean export format for systematic reviews

**How it works:** 
1. Uploads are converted to structured markdown using Marker + LLM (highest accuracy mode)
2. AI agents use markdown structure to navigate sections and extract data more accurately
3. Fallback to pdfplumber if Marker is unavailable
4. LLM enhancement improves complex table parsing and structure recognition
""")

uploaded_file = st.file_uploader("Upload a PDF clinical trial report to begin", type="pdf")

if uploaded_file is not None:
    file_contents = uploaded_file.getvalue()
    
    # Use enhanced markdown conversion
    markdown_text = convert_pdf_to_markdown_with_marker(file_contents)
    
    if markdown_text:
        # Show markdown preview option
        with st.expander("👁️ Preview Converted Markdown (First 2000 characters)"):
            st.text(markdown_text[:2000] + "..." if len(markdown_text) > 2000 else markdown_text)
        
        study_metadata, outcomes = run_enhanced_extraction_pipeline(markdown_text)

        if outcomes:
            st.success(f"✅ Enhanced processing complete for **{uploaded_file.name}**")
            
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
            
            # Display outcomes with hierarchical structure and details
            st.subheader("🎯 Extracted Outcomes with Details")
            df = pd.DataFrame(outcomes)
            for col in ['outcome_domain', 'outcome_specific', 'outcome_type', 'definition', 'measurement_method', 'timepoint']:
                if col not in df.columns: 
                    df[col] = ''
            df.fillna('', inplace=True)

            # HIERARCHICAL DISPLAY WITH DETAILS
            domains = df[df['outcome_domain'] != '']['outcome_domain'].unique()
            for domain in domains:
                st.markdown(f"**DOMAIN:** {domain}")
                
                # Show domain-level details if available
                domain_row = df[df['outcome_domain'] == domain].iloc[0]
                if domain_row.get('definition'):
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Definition:* {domain_row['definition']}")
                if domain_row.get('timepoint'):
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Timepoint:* {domain_row['timepoint']}")
                
                # Show specific outcomes
                specific_outcomes = df[(df['outcome_domain'] == domain) & (df['outcome_specific'] != '') & (df['outcome_specific'] != domain)]
                if len(specific_outcomes) > 0:
                    for _, outcome in specific_outcomes.iterrows():
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• **{outcome['outcome_specific']}**")
                        if outcome.get('definition'):
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Definition:* {outcome['definition']}")
                        if outcome.get('measurement_method'):
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Method:* {outcome['measurement_method']}")
                        if outcome.get('timepoint'):
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Timepoint:* {outcome['timepoint']}")
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
                    "Measurement_Method": "",
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
                    "Measurement_Method": domain_row.get('measurement_method', ''),
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
                        "Measurement_Method": specific_row.get('measurement_method', ''),
                        "Timepoint": specific_row.get('timepoint', '')
                    }
                    export_rows.append(specific_outcome_row)
            
            export_df = pd.DataFrame(export_rows)

            st.download_button(
                label="📥 **Download Complete Dataset (CSV)**",
                data=export_df.to_csv(index=False).encode('utf-8'),
                file_name=f"Enhanced_Study_Data_{uploaded_file.name.replace('.pdf', '')}.csv",
                mime='text/csv',
                help="Enhanced dataset: Study metadata + all outcomes with definitions, methods & timepoints"
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

# ---------- 6. INSTALLATION INSTRUCTIONS ----------

st.sidebar.markdown("""
## 🔧 Setup Instructions

To use Marker with LLM enhancement:

```bash
pip install marker-pdf
```

**LLM Enhancement Benefits:**
- Highest accuracy conversion
- Better complex table parsing
- Improved structure recognition
- Enhanced formatting preservation

**System Requirements:**
- 4-8GB RAM recommended
- GPU optional but faster
- First run downloads ~2GB models

**Fallback:** If Marker fails, the system automatically falls back to pdfplumber.
""")