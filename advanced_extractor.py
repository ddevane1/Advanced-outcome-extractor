#!/usr/bin/env python3

# -------- universal_clinical_trial_extractor.py (v12.0 - universal extraction with adverse events) --------

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

# ---------- 2. UNIVERSAL SPECIALIZED AGENT FUNCTIONS ----------

def agent_extract_comprehensive_metadata(full_text: str) -> dict:
    """Agent 1: Extracts comprehensive study metadata from ANY clinical trial paper."""
    prompt = f'''You are a universal clinical trial metadata extraction specialist. From this document, extract ALL available study information. This works for ANY clinical trial paper format (NEJM, Lancet, JAMA, etc.). If any value is not found, use null.

**REQUIRED EXTRACTION FIELDS:**
- authors: Complete author list as written
- first_author_surname: Just the surname of the first author
- publication_year: Year published
- journal: Journal name
- study_design: Type of study (RCT, cohort, case-control, observational, systematic review, etc.)
- study_country: Country/countries where study was conducted
- patient_population: Description of study participants
- targeted_condition: Disease/condition being studied
- diagnostic_criteria: Criteria used to diagnose the condition
- interventions_tested: All interventions/treatments tested
- comparison_group: Control group or comparator details

**SEARCH STRATEGY:**
- Look in title page, abstract, methods section, author affiliations
- For authors: Check bylines, author lists, affiliations
- For study design: Check abstract, methods, or title
- For interventions: Check methods, abstract, results
- For population: Check methods, participants section
- For country: Check author affiliations, methods, or study sites

Respond in this exact JSON format:
{{
  "study_metadata": {{
    "authors": "...",
    "first_author_surname": "...",
    "publication_year": "...",
    "journal": "...",
    "study_design": "...",
    "study_country": "...",
    "patient_population": "...",
    "targeted_condition": "...",
    "diagnostic_criteria": "...",
    "interventions_tested": "...",
    "comparison_group": "..."
  }}
}}

**Document Text to Analyze (first 10,000 characters):**
{full_text[:10000]}'''

    return parse_json_response(ask_llm(prompt), "study_metadata")

def agent_locate_all_outcomes(full_text: str) -> list:
    """Agent 2: Finds ALL outcomes from ANY clinical trial paper - primary, secondary, safety, etc."""
    prompt = f'''You are a universal clinical trial outcome extraction specialist. Extract ALL outcomes from this document, regardless of the journal format or study type.

**WHAT TO EXTRACT:**
1. **Primary outcomes** (usually in Methods/Protocol sections)
2. **Secondary outcomes** (usually in Methods/Protocol sections)
3. **Safety outcomes** (may be in Methods or Safety sections)
4. **Exploratory outcomes** (if mentioned)
5. **Post-hoc outcomes** (if mentioned)

**EXTRACTION RULES:**
1. **Multiple Formats:** Handle outcomes presented as:
   - Numbered lists (1. Primary outcome: ...)
   - Bullet points
   - Paragraph descriptions
   - Table headers that describe outcomes
2. **Time-Based Variations:** If same outcome measured at different timepoints, create separate entries
3. **Composite Outcomes:** If an outcome has multiple components, extract both the composite and individual components

**OUTPUT FORMAT:** Return JSON with 'defined_outcomes' list. Each item needs:
{{
  "outcome_domain": "Main outcome category",
  "outcome_specific": "Specific measure if applicable", 
  "outcome_type": "primary/secondary/safety/exploratory/post-hoc",
  "definition": "How outcome is defined",
  "measurement_method": "How it's measured/assessed",
  "timepoint": "When measured"
}}

**Document Text to Analyze:**
{full_text}'''

    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "defined_outcomes") or []

def agent_parse_universal_table(table_text: str) -> list:
    """Agent 3: Universal table parser that captures TRUE hierarchical structure from tables."""
    prompt = f'''You are an expert at parsing clinical trial tables with PERFECT hierarchical structure recognition.

**CRITICAL UNDERSTANDING:**
In clinical trial tables, the structure is typically:
- **MAIN SECTION HEADERS** (often bold/caps) = outcome_domain
- **INDENTED SUB-ITEMS** underneath = outcome_specific

**STEP 1: CLASSIFY THE TABLE**
Determine the table type:
- **Baseline/Demographics:** Patient characteristics, demographics, medical history
- **Primary/Secondary Outcomes:** Main study results, efficacy data  
- **Adverse Events/Safety:** Side effects, complications, safety events
- **Subgroup Analysis:** Results by patient subgroups
- **Other:** Laboratory values, pharmacokinetics, etc.

**STEP 2: EXTRACT TRUE HIERARCHICAL STRUCTURE**

**If Baseline/Demographics table:** Return `{{"table_outcomes": []}}`

**If ANY other table type:** Extract with these PRECISE rules:

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

5. **SMART TIMEPOINT EXTRACTION:**
   - Extract timepoints from domain names: "Adverse outcomes at <37 wk of gestation" → timepoint: "<37 wk of gestation"
   - Clean outcome names but preserve all qualifiers

**OUTPUT FORMAT:** JSON with 'table_outcomes' list:
{{
  "outcome_domain": "Exact main section header (timepoint extracted if applicable)",
  "outcome_specific": "Exact sub-item text (complete with all qualifiers)",
  "outcome_type": "safety/efficacy/laboratory/other",
  "definition": "...",
  "measurement_method": "...",
  "timepoint": "Extracted timepoint or from domain"
}}

**TABLE TEXT TO PARSE:**
{table_text}'''

    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "table_outcomes") or []

def agent_finalize_comprehensive_structure(messy_list: list) -> list:
    """Agent 4: Enhanced structuring that PRESERVES exact hierarchical relationships and complete names."""
    prompt = f'''You are a clinical trial data structuring expert. Your job is to clean and organize while PRESERVING the exact hierarchical structure and complete outcome names.

**CRITICAL PRESERVATION RULES:**

1. **PRESERVE EXACT HIERARCHICAL STRUCTURE:**
   - Keep domain-specific relationships EXACTLY as extracted
   - "Adverse outcomes at <37 wk of gestation" should remain as domain with its specific sub-outcomes
   - "Stillbirth or death" should remain as domain with its specific sub-outcomes
   - DO NOT create generic domains like "Adverse outcomes" or "Neonatal outcomes"

2. **PRESERVE COMPLETE OUTCOME NAMES:**
   - "Miscarriage or stillbirth without preeclampsia" must stay EXACTLY as is
   - "Small-for-gestational-age status without preeclampsia" must stay EXACTLY as is
   - "With preeclampsia or status of being small for gestational age" must stay EXACTLY as is
   - DO NOT truncate or simplify these names

3. **MINIMAL CLEANING ONLY:**
   - Only remove obvious formatting artifacts like "— no. (%)" 
   - Only standardize obvious duplicates (same domain AND same specific)
   - DO NOT change or simplify outcome names
   - DO NOT merge different domains together

4. **PRESERVE TIMEPOINT INFORMATION:**
   - Keep timepoints from domain names: "at <37 wk of gestation" 
   - Inherit domain timepoints to specific outcomes if they don't have their own
   - Standardize format: "<37 wk" → "<37 weeks of gestation"

5. **OUTCOME TYPE ASSIGNMENT:**
   - "primary" for primary endpoints
   - "secondary" for secondary endpoints  
   - "safety" for adverse events, side effects, safety outcomes
   - "efficacy" for treatment effectiveness measures
   - "exploratory" for exploratory/post-hoc analyses
   - "other" for miscellaneous outcomes

**FORBIDDEN ACTIONS:**
- DO NOT create generic domain names
- DO NOT truncate outcome names 
- DO NOT merge different domains
- DO NOT simplify medical terminology

**OUTPUT STRUCTURE:** Each outcome gets its own row preserving EXACT structure:
{{
  "outcome_type": "primary/secondary/safety/efficacy/exploratory/other",
  "outcome_domain": "EXACT domain name as extracted",
  "outcome_specific": "EXACT specific name as extracted", 
  "definition": "How outcome is defined (consolidated from multiple sources)",
  "measurement_method": "How measured (consolidated)",
  "timepoint": "When measured (inherited from domain if applicable)"
}}

**EXAMPLE OF CORRECT PRESERVATION:**
Input data with domain "Adverse outcomes at <37 wk of gestation" and specific "Miscarriage or stillbirth without preeclampsia"

Output: 
- outcome_domain: "Adverse outcomes at <37 wk of gestation"
- outcome_specific: "Miscarriage or stillbirth without preeclampsia"
- timepoint: "<37 weeks of gestation"

**MESSY LIST TO PROCESS:**
{json.dumps(messy_list, indent=2)}'''

    return parse_json_response(ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE), "final_outcomes") or []

# ---------- 3. UNIVERSAL TABLE DETECTION ----------

def extract_all_tables(full_text: str) -> list:
    """Enhanced universal table extraction with robust deduplication."""
    # Multiple patterns to catch different table formats
    patterns = [
        r"(Table \d+\..*?)(?=\nTable \d+\.|\nFigure \d+\.|\n\n[A-Z][A-Z\s]+\n|\Z)",  # Standard format
        r"(TABLE \d+\..*?)(?=\nTABLE \d+\.|\nFIGURE \d+\.|\n\n[A-Z][A-Z\s]+\n|\Z)",  # All caps
        r"(\nTable \d+[:\.].*?)(?=\nTable \d+|\nFigure \d+|\n\n[A-Z][A-Z\s]+\n|\Z)",  # Slight variation
        r"(\n\s*Table \d+.*?)(?=\n\s*Table \d+|\n\s*Figure \d+|\n\n[A-Z][A-Z\s]+\n|\Z)"  # With spaces
    ]
    
    all_tables = []
    for pattern in patterns:
        tables = re.findall(pattern, full_text, re.DOTALL | re.IGNORECASE)
        all_tables.extend(tables)
    
    # Enhanced deduplication with multiple signature methods
    seen_signatures = set()
    unique_tables = []
    
    for table in all_tables:
        # Create multiple signatures to catch edge cases
        signatures = [
            # Original method: first 100 chars
            re.sub(r'\s+', ' ', table[:100]),
            # Table number + first line
            re.sub(r'\s+', ' ', table[:200].split('\n')[0]) if '\n' in table[:200] else table[:100],
            # Content-based: first few data rows (skip title)
            re.sub(r'\s+', ' ', '\n'.join(table.split('\n')[2:4])) if len(table.split('\n')) > 3 else table[:100]
        ]
        
        # Check if any signature has been seen
        is_duplicate = any(sig in seen_signatures for sig in signatures if sig.strip())
        
        if not is_duplicate:
            # Add all signatures to seen set
            for sig in signatures:
                if sig.strip():
                    seen_signatures.add(sig)
            unique_tables.append(table)
    
    return unique_tables

# ---------- 4. MAIN UNIVERSAL ORCHESTRATION PIPELINE ----------

@st.cache_data(show_spinner="Step 2: Running comprehensive AI extraction pipeline...")
def run_universal_extraction_pipeline(full_text: str):
    """Universal orchestration that works for ANY clinical trial paper."""
    
    # Extract comprehensive metadata
    study_metadata = agent_extract_comprehensive_metadata(full_text)
    
    # Extract all defined outcomes from text
    defined_outcomes = agent_locate_all_outcomes(full_text)
    
    # Extract all tables and parse them
    table_texts = extract_all_tables(full_text)
    all_table_outcomes = []
    
    if table_texts:
        for table_text in table_texts:
            parsed_outcomes = agent_parse_universal_table(table_text)
            if parsed_outcomes:
                all_table_outcomes.extend(parsed_outcomes)
    
    # Combine all outcomes
    raw_combined_list = defined_outcomes + all_table_outcomes
    
    if not raw_combined_list:
        return study_metadata, []
    
    # Final structuring and cleaning
    final_outcomes = agent_finalize_comprehensive_structure(raw_combined_list)
    
    return study_metadata, final_outcomes

# ---------- 5. ENHANCED STREAMLIT UI ----------

st.set_page_config(layout="wide")
st.title("Universal Clinical Trial Data Extractor (v12.0)")
st.markdown("""
**✨ NEW FEATURES:**
- **Universal:** Works with ANY clinical trial paper format (NEJM, Lancet, JAMA, etc.)
- **Comprehensive:** Extracts complete study metadata + ALL outcomes
- **Adverse Events:** Captures safety outcomes and adverse events from any table
- **No Duplication:** Study info shown once, outcomes listed separately
- **Publication Ready:** Clean export format for systematic reviews
""")

uploaded_file = st.file_uploader("Upload ANY clinical trial PDF to begin", type="pdf")

if uploaded_file is not None:
    file_contents = uploaded_file.getvalue()
    full_text = get_pdf_text(file_contents)
    
    if full_text:
        study_metadata, outcomes = run_universal_extraction_pipeline(full_text)
        
        if outcomes:
            st.success(f"✅ Extraction complete for **{uploaded_file.name}**")
            
            # Display study metadata
            st.subheader("📋 Study Information")
            if study_metadata:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Authors:** {study_metadata.get('authors', 'Not found')}")
                    st.write(f"**Journal:** {study_metadata.get('journal', 'Not found')}")
                    st.write(f"**Year:** {study_metadata.get('publication_year', 'Not found')}")
                    st.write(f"**Study Design:** {study_metadata.get('study_design', 'Not found')}")
                    st.write(f"**Country:** {study_metadata.get('study_country', 'Not found')}")
                
                with col2:
                    st.write(f"**Population:** {study_metadata.get('patient_population', 'Not found')}")
                    st.write(f"**Condition:** {study_metadata.get('targeted_condition', 'Not found')}")
                    st.write(f"**Interventions:** {study_metadata.get('interventions_tested', 'Not found')}")
                    st.write(f"**Comparison:** {study_metadata.get('comparison_group', 'Not found')}")
            
            # Display outcomes with proper hierarchical structure
            st.subheader("🎯 Extracted Outcomes")
            
            df = pd.DataFrame(outcomes)
            
            # Organize by outcome type
            outcome_types = ['primary', 'secondary', 'safety', 'efficacy', 'exploratory', 'other']
            
            for outcome_type in outcome_types:
                type_outcomes = df[df['outcome_type'] == outcome_type] if 'outcome_type' in df.columns else pd.DataFrame()
                
                if not type_outcomes.empty:
                    st.markdown(f"**{outcome_type.upper()} OUTCOMES**")
                    
                    # Group by domain to show hierarchical structure
                    domains = type_outcomes['outcome_domain'].unique()
                    
                    for domain in domains:
                        domain_outcomes = type_outcomes[type_outcomes['outcome_domain'] == domain]
                        
                        # Show domain
                        st.markdown(f"• **{domain}**")
                        
                        # Show specific outcomes under this domain
                        for _, outcome in domain_outcomes.iterrows():
                            specific = outcome.get('outcome_specific', '')
                            if specific and specific.strip():
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;◦ {specific}")
                            # If no specific, the domain itself is the complete outcome
                    
                    st.write("")
            
            # COMPREHENSIVE EXPORT
            st.subheader("📊 Export Data")
            
            # Create comprehensive export with NO DUPLICATION
            export_data = []
            
            # Add study metadata once at the top
            base_row = {
                "Authors": study_metadata.get('authors', '') if study_metadata else '',
                "First_Author_Surname": study_metadata.get('first_author_surname', '') if study_metadata else '',
                "Publication_Year": study_metadata.get('publication_year', '') if study_metadata else '',
                "Journal": study_metadata.get('journal', '') if study_metadata else '',
                "Study_Design": study_metadata.get('study_design', '') if study_metadata else '',
                "Study_Country": study_metadata.get('study_country', '') if study_metadata else '',
                "Patient_Population": study_metadata.get('patient_population', '') if study_metadata else '',
                "Targeted_Condition": study_metadata.get('targeted_condition', '') if study_metadata else '',
                "Diagnostic_Criteria": study_metadata.get('diagnostic_criteria', '') if study_metadata else '',
                "Interventions_Tested": study_metadata.get('interventions_tested', '') if study_metadata else '',
                "Comparison_Group": study_metadata.get('comparison_group', '') if study_metadata else '',
            }
            
            # Add each outcome as a separate row (study info will be same for all)
            for _, outcome in df.iterrows():
                row = base_row.copy()  # Study info stays constant
                row.update({
                    "Outcome_Type": outcome.get('outcome_type', ''),
                    "Outcome_Domain": outcome.get('outcome_domain', ''),
                    "Outcome_Specific": outcome.get('outcome_specific', ''),
                    "Definition": outcome.get('definition', ''),
                    "Measurement_Method": outcome.get('measurement_method', ''),
                    "Timepoint": outcome.get('timepoint', '')
                })
                export_data.append(row)
            
            export_df = pd.DataFrame(export_data)
            
            # Download button
            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 **Download Complete Dataset (CSV)**",
                data=csv_data,
                file_name=f"Clinical_Trial_Data_{uploaded_file.name.replace('.pdf', '')}.csv",
                mime='text/csv',
                help="Complete dataset: Study metadata + all outcomes. Study info constant, outcomes vary per row."
            )
            
            # Show preview of export
            with st.expander("👁️ Preview Export Data"):
                st.dataframe(export_df.head(10))
                st.info(f"Total rows: {len(export_df)} (1 row per outcome, study metadata repeated)")
        
        else:
            st.warning("⚠️ No outcomes were extracted. The document may not contain recognizable outcome data.")
            
            # Still show metadata if available
            if study_metadata:
                st.subheader("📋 Study Metadata Found")
                for key, value in study_metadata.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")