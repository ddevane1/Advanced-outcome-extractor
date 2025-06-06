#!/usr/bin/env python3
# -------- advanced_extractor.py (v16.2 – patched, stable) --------
"""
Clinical-Trial Outcome Extractor
Patch notes v16.2
- Fixed TypeError on list concatenation by adding a defensive check.
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

client = OpenAI(
    api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
)

# ---------- 1. CORE HELPER FUNCTIONS ----------


@st.cache_data
def get_pdf_text(file_contents: bytes) -> str | None:
    """Extract text from PDF bytes and cache the result."""
    st.info("Step 1 / 3 – reading PDF text …")
    try:
        with pdfplumber.open(io.BytesIO(file_contents)) as pdf:
            full_text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)

        if not full_text.strip():
            st.error("This PDF appears to be scanned images only – no extractable text.")
            return None

        st.success("✓ PDF text read successfully")
        return full_text

    except Exception as exc:  # broad except okay here – we want to surface to UI
        st.error(f"Error reading PDF: {exc}")
        return None


def ask_llm(prompt: str, max_response_tokens: int = DEFAULT_TOKENS_FOR_RESPONSE) -> str | None:
    """Generic function to call the OpenAI chat API in JSON-object mode."""

    final_prompt = (
        "You must provide a response in a JSON object. "
        + prompt
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.0,
            max_tokens=max_response_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as exc:
        st.error(f"OpenAI API error: {exc}")
        return None


def parse_json_response(response_text: str | None, key: str | None):
    """Safely parse JSON and return either the whole dict or a key within it."""
    if not response_text:
        return None
    try:
        data = json.loads(response_text)
        return data if key is None else data.get(key)
    except (json.JSONDecodeError, AttributeError):
        st.warning("Could not parse valid JSON from AI response.")
        return None


# ---------- 2. SPECIALISED AGENT FUNCTIONS ----------


def agent_extract_metadata(full_text: str) -> dict | None:
    """Agent 1 – extract high-level study metadata."""

    prompt = (
        f"""From the beginning of this document, extract the study information. If a value
        is absent, use null. Respond with a key \"study_info\".\n\nText to analyse:\n{full_text[:8000]}"""
    )

    return parse_json_response(ask_llm(prompt), "study_info")


def agent_locate_defined_outcomes(full_text: str) -> list:
    """Agent 2 – locate planned outcomes described in the Methods section."""

    prompt = (
        f"""Extract all outcome definitions (typically found in the Methods section).
        Handle semicolon-separated lists and time-based grouping as separate domains.
        Respond with a list called \"defined_outcomes\".\n\nDocument text to analyse:\n{full_text}"""
    )

    return parse_json_response(ask_llm(prompt), "defined_outcomes") or []


def agent_parse_table(table_text: str) -> list:
    """Agent 3 – parse a single table and extract outcome names (if outcome table)."""

    prompt = (
        f"""Analyse the single table text below. First, classify it as BASELINE or OUTCOME.
        If BASELINE, return an empty list. If OUTCOME, extract the clean outcome names,
        distinguishing between domains and specific outcomes. Strip away all data columns.
        Respond with a list called \"table_outcomes\".\n\nTABLE TEXT TO PARSE:\n{table_text}"""
    )

    return parse_json_response(ask_llm(prompt), "table_outcomes") or []


def agent_finalize_and_structure(messy_list: list) -> list:
    """Agent 4 – clean, dedupe and structure the combined outcome list."""

    prompt = (
        f"""Clean, deduplicate and structure this messy list of outcomes into a final
        hierarchical list. Create \"domain\" and \"specific\" types. Respond with a key
        \"final_outcomes\".\n\nMESSY LIST TO PROCESS:\n{json.dumps(messy_list, indent=2)}"""
    )

    return parse_json_response(
        ask_llm(prompt, max_response_tokens=LARGE_TOKENS_FOR_RESPONSE),
        "final_outcomes",
    ) or []


# ---------- 3. MAIN ORCHESTRATION PIPELINE ----------


@st.cache_data(show_spinner="Step 2 / 3 – running AI extraction pipeline …")
def run_extraction_pipeline(full_text: str):
    """Orchestrate the calls to the four specialist agents."""

    study_info = agent_extract_metadata(full_text)
    defined_outcomes = agent_locate_defined_outcomes(full_text)

    # Extract consecutive blocks that start with "Table X." up to the next table or EOF
    table_texts = re.findall(r"(Table \d+\..*?)(?=\nTable \d+\.|\Z)", full_text, re.DOTALL)

    all_table_outcomes: list = []
    for table_text in table_texts:
        parsed_outcomes = agent_parse_table(table_text)
        if parsed_outcomes: # Ensure we only extend if the result is not None/empty
            all_table_outcomes.extend(parsed_outcomes)

    # --- THIS IS THE FIX ---
    # Defensively ensure both operands are lists before concatenation
    safe_defined_outcomes = defined_outcomes if isinstance(defined_outcomes, list) else []
    safe_table_outcomes = all_table_outcomes if isinstance(all_table_outcomes, list) else []
    raw_combined_list = safe_defined_outcomes + safe_table_outcomes
    # -----------------------

    if not raw_combined_list:
        # Nothing → nothing: return metadata and empty outcomes list
        return study_info, []

    final_outcomes = agent_finalize_and_structure(raw_combined_list)
    return study_info, final_outcomes


# ---------- 4. STREAMLIT UI ----------


st.set_page_config(layout="wide", page_title="Clinical Trial Outcome Extractor v16.2")
st.title("Clinical-Trial Outcome Extractor (v16.2)")
st.markdown(
    "This tool uses a cached, multi-agent AI workflow to accurately and reliably "
    "extract outcomes from PDF clinical-trial reports."
)

uploaded_file = st.file_uploader(
    "Upload a PDF clinical-trial report to begin", type="pdf", accept_multiple_files=False
)

if uploaded_file is not None:
    file_contents = uploaded_file.getvalue()
    full_text = get_pdf_text(file_contents)

    if full_text:
        study_info, outcomes = run_extraction_pipeline(full_text)

        if outcomes:
            st.success(f"Processing complete for **{uploaded_file.name}**.")

            # --- tidy outcomes into a dataframe ---
            df = pd.DataFrame(outcomes)
            for col in [
                "outcome_domain",
                "outcome_specific",
                "outcome_type",
                "definition",
                "timepoint",
            ]:
                if col not in df.columns:
                    df[col] = ""
            df.fillna("", inplace=True)

            # --- hierarchical pretty print ---
            st.subheader("Hierarchical Outcome View")
            domains = (
                df[df["outcome_domain"].astype(str) != ""]["outcome_domain"].unique()
            )
            for domain in domains:
                st.markdown(f"**DOMAIN:** {domain}")
                specific_outcomes = df[
                    (df["outcome_domain"] == domain)
                    & (df["outcome_specific"] != "")
                    & (df["outcome_specific"] != domain)
                ]["outcome_specific"].unique()

                if specific_outcomes.size:
                    for specific in specific_outcomes:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;• {specific}")
                else:
                    st.markdown(
                        "&nbsp;&nbsp;&nbsp;• *Primary outcome or domain with no specific sub-outcomes.*"
                    )

                st.write("")  # blank line

            # --- export CSV ---
            st.subheader("Export Results")
            export_rows: list[dict[str, str]] = []
            for domain in domains:
                domain_df = df[df["outcome_domain"] == domain]
                if domain_df.empty:
                    continue
                domain_row = domain_df.iloc[0]
                export_rows.append(
                    {
                        "Domain": domain,
                        "Specific Outcome": "",
                        "Definition": domain_row.get("definition", ""),
                        "Timepoint": domain_row.get("timepoint", ""),
                    }
                )

                specific_df = df[
                    (df["outcome_domain"] == domain)
                    & (df["outcome_specific"] != "")
                    & (df["outcome_specific"] != domain)
                ]
                for _, row in specific_df.iterrows():
                    export_rows.append(
                        {
                            "Domain": "",
                            "Specific Outcome": row["outcome_specific"],
                            "Definition": row.get("definition", ""),
                            "Timepoint": row.get("timepoint", ""),
                        }
                    )

            export_df = pd.DataFrame(export_rows)

            st.download_button(
                label="**Download Publication-Ready CSV**",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"Publication_Outcomes_{uploaded_file.name}.csv",
                mime="text/csv",
            )

            # --- side info ---
            with st.expander("Show Extracted Study Information"):
                st.json(study_info or {})

            with st.expander("Show Full Raw Data Table (for analysis)"):
                st.dataframe(df.astype(str))

        else:
            st.error("Extraction ran but *no* outcomes were found.")