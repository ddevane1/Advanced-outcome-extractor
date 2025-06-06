#!/usr/bin/env python3

# -------- enhanced_clinical_trial_extractor.py (v12.0 - comprehensive with timepoints & definitions) --------

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

def agent_extract_results_outcomes(full_text: str) -> list:
    """Agent 2b: Extracts outcomes specifically mentioned in Results/Follow-up sections."""
    prompt = f'''You are a clinical trial results analyst. Scan this document for outcomes that are REPORTED in Results, Follow-up, or Discussion sections but may not be formally defined in Methods.

**CRITICAL MISSION:**
Find outcomes that are actually measured and reported, even if not pre-specified as formal endpoints.

**WHAT TO LOOK FOR:**
- **Death/Mortality:** Any mention of deaths, mortality, fatalities
- **Censures:** Patients lost to follow-up, withdrawn, censored
- **Adverse Events:** Side effects, complications mentioned in results
- **Hospital outcomes:** Length of stay, readmissions, etc.
- **Post-hoc outcomes:** Outcomes analyzed after the fact

**EXAMPLES FROM RESULTS SECTIONS:**
- "During follow-up, there were 30 deaths none due to RSV infection"
  → domain: "Death", definition: "mortality during follow-up", timepoint: "during follow-up"
- "1,536 censures occurred during the study period"
  → domain: "Censures", definition: "patients lost to follow-up", timepoint: "during study period"
- "5 patients had severe adverse reactions requiring hospitalization"
  → domain: "Severe adverse reactions", definition: "reactions requiring hospitalization"

**SEARCH STRATEGY:**
- Focus on Results, Follow-up, Safety, and Discussion sections
- Look for phrases like: "there were X deaths", "Y patients died", "mortality occurred", "adverse events included"
- Capture cause-specific outcomes if mentioned
- Note timing/follow-up periods

**OUTPUT FORMAT:** Return JSON with 'results_outcomes' list:
{{
  "outcome_domain": "Main outcome name",
  "outcome_specific": "Specific measure if applicable", 
  "definition": "HOW outcome was identified/measured",
  "measurement_method": "How it was assessed",
  "timepoint": "WHEN it was measured/occurred"
}}

**Document Text to Analyze:**
{full_text}'''

    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "results_outcomes") or []

def agent_locate_defined_outcomes_with_details(full_text: str) -> list:
    """Agent 2: Finds planned outcomes with comprehensive definitions and timepoints."""
    prompt = f'''You are a clinical trial protocol analyst. Extract all outcome definitions from this document with COMPLETE details including definitions and timepoints.

**CRITICAL EXTRACTION RULES:**
1. **Extract Complete Definitions:** Capture HOW each outcome is defined/measured
2. **Extract Timepoints:** Capture WHEN each outcome is measured/assessed
3. **Extract Measurement Methods:** Capture the instruments/scales/criteria used
4. **Handle Lists:** Treat semicolon-separated items as separate outcomes
5. **INCLUDE FOLLOW-UP OUTCOMES:** Also capture outcomes mentioned in Results/Follow-up sections

**WHAT TO LOOK FOR:**
- Primary outcome definitions and timepoints
- Secondary outcome definitions and timepoints  
- Safety outcome definitions and timepoints
- **Follow-up outcomes:** Death, mortality, censures, adverse events mentioned in Results
- Measurement instruments (scales, questionnaires, lab tests)
- Assessment timepoints (baseline, 30 days, discharge, follow-up, etc.)

**EXAMPLES:**
- "Primary outcome was delivery with preeclampsia before 37 weeks of gestation" 
  → domain: "Preeclampsia", definition: "delivery with preeclampsia", timepoint: "before 37 weeks of gestation"
- "During follow-up, there were 30 deaths none due to RSV infection"
  → domain: "Death", definition: "mortality during follow-up", timepoint: "during follow-up"
- "Secondary outcomes were adverse outcomes before 34 weeks, before 37 weeks, and at or after 37 weeks of gestation"
  → Extract as separate outcomes with different timepoints

**SEARCH LOCATIONS:**
- Methods/Outcome Measures sections
- Results sections (for reported outcomes like death, censures)
- Follow-up sections
- Discussion of outcomes achieved

**OUTPUT FORMAT:** Return JSON with 'defined_outcomes' list:
{{
  "outcome_domain": "Main outcome name",
  "outcome_specific": "Specific measure if applicable",
  "definition": "HOW the outcome is defined/measured",
  "measurement_method": "Instrument/scale/criteria used",
  "timepoint": "WHEN outcome is measured/assessed"
}}

**Document Text to Analyze:**
{full_text}'''

    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "defined_outcomes") or []

def agent_parse_table_enhanced_with_details(table_text: str) -> list:
    """Agent 3: Enhanced table parser that captures hierarchical structure, complete names, definitions and timepoints."""
    prompt = f'''You are an expert at parsing clinical trial tables with PERFECT extraction of hierarchical structure, definitions, and timepoints.

**STEP 1: CLASSIFY THE TABLE**
First, determine if this table describes **baseline patient characteristics** (demographics, age, etc.) or **clinical trial outcomes** (results, events, complications, adverse events).

**STEP 2: EXTRACT WITH COMPREHENSIVE DETAIL**

**If BASELINE table:** Return `{{"table_outcomes": []}}`

**If OUTCOME table:** Extract with these PRECISE rules:

1. **IDENTIFY HIERARCHICAL STRUCTURE:**
   ```
   Main Header (often bold/caps)               <- This is outcome_domain
       Sub-item 1 — no. (%)                   <- This is outcome_specific  
       Sub-item 2 — no. (%)                   <- This is outcome_specific
   ```

2. **PRESERVE COMPLETE NAMES INCLUDING ALL QUALIFIERS:**
   - "Miscarriage or stillbirth without preeclampsia — no. (%)" becomes:
     * outcome_specific: "Miscarriage or stillbirth without preeclampsia"
   - "Respiratory distress syndrome treated with surfactant and ventilation — no. (%)" becomes:
     * outcome_specific: "Respiratory distress syndrome treated with surfactant and ventilation"

3. **EXTRACT TIMEPOINTS FROM CONTEXT:**
   - From headers like "Adverse outcomes at <37 wk of gestation" → timepoint: "<37 wk of gestation"
   - From headers like "Secondary outcomes before 34 weeks" → timepoint: "before 34 weeks"
   - From outcome names like "Primary endpoint at 30 days" → timepoint: "30 days"

4. **EXTRACT DEFINITIONS WHERE AVAILABLE:**
   - Look for explanatory text about how outcomes are defined
   - Capture measurement criteria mentioned in table footnotes or headers
   - Note any diagnostic criteria referenced

5. **EXAMPLES OF CORRECT EXTRACTION:**
   From table text like:
   ```
   Adverse outcomes at <37 wk of gestation
       Preeclampsia — no. (%)                 13 (1.6)    35 (4.3)
       Gestational hypertension — no. (%)     8 (1.0)     7 (0.9)
       Small-for-gestational-age status without preeclampsia — no./total no. (%)  17/785 (2.2)  18/807 (2.2)
   ```
   
   Should extract:
   - domain="Adverse outcomes", specific="Preeclampsia", timepoint="<37 wk of gestation"
   - domain="Adverse outcomes", specific="Gestational hypertension", timepoint="<37 wk of gestation"  
   - domain="Adverse outcomes", specific="Small-for-gestational-age status without preeclampsia", timepoint="<37 wk of gestation"

**OUTPUT FORMAT:** Return JSON with 'table_outcomes' list:
{{
  "outcome_domain": "Exact main section header (timepoint extracted if applicable)",
  "outcome_specific": "Exact sub-item text (complete with all qualifiers)",
  "definition": "How outcome is defined/measured (from context)",
  "measurement_method": "Instrument/scale/criteria used (if mentioned)",
  "timepoint": "When outcome is measured/assessed"
}}

**TABLE TEXT TO PARSE:**
{table_text}'''

    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "table_outcomes") or []

def agent_finalize_comprehensive_structure(messy_list: list) -> list:
    """Agent 4: Enhanced structuring that preserves hierarchical structure, complete names, definitions and timepoints."""
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

# ---------- 3. MAIN ORCHESTRATION PIPELINE (CACHED) ----------

@st.cache_data(show_spinner="Step 2: Running comprehensive AI extraction pipeline...")
def run_comprehensive_extraction_pipeline(full_text: str):
    """Orchestrates the AI agent calls for comprehensive extraction."""
    
    # Extract comprehensive study metadata
    study_metadata = agent_extract_comprehensive_study_metadata(full_text)
    
    # Extract defined outcomes with details
    defined_outcomes = agent_locate_defined_outcomes_with_details(full_text)
    
    # Extract outcomes from Results/Follow-up sections
    results_outcomes = agent_extract_results_outcomes(full_text)
    
    # Extract and parse all tables
    table_texts = re.findall(r"(Table \d+\..*?)(?=\nTable \d+\.|\Z)", full_text, re.DOTALL)
    all_table_outcomes = []
    if table_texts:
        for table_text in table_texts:
            parsed_outcomes = agent_parse_table_enhanced_with_details(table_text)
            if parsed_outcomes:
                all_table_outcomes.extend(parsed_outcomes)
    
    # Combine all outcomes from multiple sources
    raw_combined_list = defined_outcomes + results_outcomes + all_table_outcomes
    if not raw_combined_list:
        return study_metadata, []

    # Final structuring
    final_outcomes = agent_finalize_comprehensive_structure(raw_combined_list)
    return study_metadata, final_outcomes

# ---------- 4. ENHANCED STREAMLIT UI ----------

st.set_page_config(layout="wide")
st.title("Enhanced Clinical Trial Data Extractor (v12.0)")
st.markdown("""
**✨ ENHANCED FEATURES:**
- **Comprehensive Study Metadata:** Extracts last author+year, title, journal, healthcare setting, country, etc.
- **Detailed Outcomes:** Captures definitions, measurement methods, and timepoints for all outcomes
- **Hierarchical Structure:** Preserves exact domain-specific structure (e.g., adverse events → specific events)
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
                file_name=f"Complete_Study_Data_{uploaded_file.name.replace('.pdf', '')}.csv",
                mime='text/csv',
                help="Complete dataset: Study metadata + all outcomes with definitions, methods & timepoints"
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