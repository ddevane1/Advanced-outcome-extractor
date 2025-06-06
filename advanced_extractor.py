#!/usr/bin/env python3

"""
Clinical‑Trial Outcome Extractor – **v11.3 (stable)**
────────────────────────────────────────────────────
A clean, self‑contained roll‑forward of the original v11.0 script with:

* **Study‑metadata rows** at the top of the CSV (per request).
* **OpenAI JSON safeguard** – adds a trivial system message containing the word
  “json” so `response_format={"type":"json_object"}` never raises HTTP 400.
* **Robust table pattern** – stops at *Table*, *Figure*, or end‑of‑file.
* **Dict‑vs‑list guard** – converts a dict to a list so `defined_outcomes + tbl`
  never crashes.
* **Safe DataFrame display** – ensures `outcome_domain` column exists before
  `groupby`.

No other logic was altered.
"""

from __future__ import annotations

import io
import json
import os
import re
from typing import List, Sequence

import pdfplumber
import pandas as pd
import streamlit as st
from openai import OpenAI

# ───────────────────────── CONFIG ─────────────────────────
MODEL = "gpt-4o"
DEFAULT_TOKENS = 4096
LARGE_TOKENS = 8192
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))

# ──────────────────────── HELPERS ────────────────────────

@st.cache_data
def get_pdf_text(file_bytes: bytes) -> str | None:
    """Return extracted text from a PDF or None on failure."""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as exc:  # noqa: BLE001
        st.error(f"PDF read error: {exc}")
        return None


def ask_llm(prompt: str, *, json_mode: bool = True, max_tokens: int = DEFAULT_TOKENS) -> str | None:
    """Send a prompt to the OpenAI chat API and return raw text."""
    messages = [
        {
            "role": "system",
            "content": "You are a json‑only assistant. Always answer with valid JSON.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        out = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"} if json_mode else {"type": "text"},
        )
        return out.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        st.error(f"OpenAI error: {exc}")
        return None


def parse_json(text: str | None, key: str | None) -> object | None:
    if not text:
        return None
    try:
        data = json.loads(text.strip().removeprefix("```json").removesuffix("```"))
        return data if key is None else data.get(key)
    except json.JSONDecodeError:
        st.warning("LLM did not return valid JSON → skipping.")
        return None

# ─────────────────────── AGENTS ──────────────────────────


def agent_extract_metadata(txt: str) -> dict | None:
    prompt = (
        "Extract these fields as JSON → first_author_surname, last_author_surname, publication_year, "
        "paper_title, journal, study_design, health_care_setting, study_country, patient_population, "
        "targeted_condition, diagnostic_criteria, interventions_tested, comparison_group. If missing, use null. "
        "Respond as {\"study_info\": {…}}.\n\nTEXT:\n" + txt[:8000]
    )
    return parse_json(ask_llm(prompt), "study_info")


def agent_locate_defined_outcomes(txt: str) -> list:
    prompt = (
        "Extract all outcome definitions from the Methods section. Treat semicolon lists as separate domains; "
        "split by timepoints. Respond as {\"defined_outcomes\": […]}.\n\nTEXT:\n" + txt
    )
    res = parse_json(ask_llm(prompt), "defined_outcomes") or []
    if isinstance(res, dict):
        res = [res]
    return res


def agent_parse_table(tbl: str) -> list:
    prompt = (
        "If the table is baseline demographics→return []. Otherwise strip trailing stats and output domain/specific pairs "
        "as {\"table_outcomes\": […]}.\n\nTABLE:\n" + tbl
    )
    return parse_json(ask_llm(prompt), "table_outcomes") or []


def agent_finalize(raw: list) -> list:
    prompt = (
        "Clean & deduplicate the list into hierarchical outcomes. Respond as {\"final_outcomes\": […]}.\n\nRAW:\n"
        + json.dumps(raw, indent=2)
    )
    return parse_json(ask_llm(prompt, max_tokens=LARGE_TOKENS), "final_outcomes") or []

# ────────────────────── PIPELINE ─────────────────────────

@st.cache_data(show_spinner="🔍 Extracting …")
def run_pipeline(txt: str) -> tuple[dict | None, list]:
    meta = agent_extract_metadata(txt)

    defined = agent_locate_defined_outcomes(txt)

    pattern = r"(Table\s+\d+\..*?)(?=\n(?:Table\s+\d+\.|Figure\s+\d+\.|$))"
    tables = re.findall(pattern, txt, flags=re.DOTALL)
    tbl_outs: list = []
    for t in tables:
        tbl_outs.extend(agent_parse_table(t))

    raw: list = defined + tbl_outs
    finals = agent_finalize(raw) if raw else []
    return meta, finals

# ───────────────────── STREAMLIT UI ─────────────────────
st.set_page_config(layout="wide")
st.title("Clinical‑Trial Outcome Extractor v11.3 (stable)")

pdf = st.file_uploader("Upload a clinical‑trial PDF", type="pdf")
if not pdf:
    st.stop()

text = get_pdf_text(pdf.getvalue())
if not text:
    st.stop()

meta, outs = run_pipeline(text)

# ──────────── DISPLAY META ───────────
if meta:
    st.subheader("Study information (parsed)")
    st.json(meta)
else:
    st.info("No study metadata extracted.")

# ──────────── DISPLAY OUTCOMES ───────
st.subheader("Extracted outcomes – hierarchical view")
if outs:
    df = pd.DataFrame(outs)
    if "outcome_domain" not in df.columns:
        df["outcome_domain"] = ""
    if "outcome_specific" not in df.columns:
        df["outcome_specific"] = ""

    for dom in df["outcome_domain"].unique():
        st.markdown(f"**DOMAIN:** {dom or '—'}")
        subs: Sequence[str] = (
            df[(df["outcome_domain"] == dom) & (df["outcome_specific"] != dom)]["outcome_specific"].unique()
        )
        for spec in subs:
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {spec}")
        st.write("")
else:
    st.info("No outcomes extracted.")

# ──────────── CSV EXPORT ─────────────
st.subheader("Publication‑ready CSV (study metadata first)")

def meta_row(label: str, value: str) -> dict:
    return {"Domain": label, "Specific Outcome": value or "", "Definition": "", "Timepoint": ""}

meta_rows: List[dict] = []
if meta:
    m = meta
    meta_rows.extend(
        [
            meta_row("Last author + year", f"{m.get('last_author_surname', '')} {m.get('publication_year', '')}"),
            meta_row("Paper title", m.get("paper_title", "")),
            meta_row("Journal", m.get("journal", "")),
            meta_row("Health‑care setting", m.get("health_care_setting", "")),
            meta_row("Country recruited", m.get("study_country", "")),
            meta_row("Patient population", m.get("patient_population", "")),
            meta_row(
                "Targeted condition (definition)",
                f"{m.get('targeted_condition', '')} ({m.get('diagnostic_criteria', '')})".strip(),
            ),
            meta_row("Intervention tested", m.get("interventions_tested", "")),
            meta_row("Comparator", m.get("comparison_group", "")),
        ]
    )

out_rows: List[dict] = []
if outs:
    for dom in df["outcome_domain"].unique():
        dom_row = df[df["outcome_domain"] == dom].iloc[0]
        out_rows.append(
            {
                "Domain": dom,
                "Specific Outcome": "",
                "Definition": dom_row.get("definition", ""),
                "Timepoint": dom_row.get("timepoint", ""),
            }
        )
        for spec in df[(df["outcome_domain"] == dom) & (df["outcome_specific"] != dom)]["outcome_specific"].unique():
            out_rows.append({"Domain": "", "Specific Outcome": spec, "Definition": "", "Timepoint": ""})

csv_df = pd.DataFrame(meta_rows + out_rows)
if not csv_df.empty:
    st.download_button(
        "Download CSV",
        csv_df.to_csv(index=False).encode(),
        file_name="clinical_trial_outcomes.csv",
        mime="text/csv",
    )
else:
    st.info("Nothing to export yet.")
