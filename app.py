import sys
import os
import re
from difflib import SequenceMatcher
from rapidfuzz import fuzz, process

import pandas as pd
import streamlit as st     # last import

# MUST be the first Streamlit command
st.set_page_config(
    page_title="Citrix Event Lookup Tool",
    layout="wide",
    page_icon="",
)

CITRIX_BLUE = "#009FD9"


# ----------------------------------------------------------
# Resource + Data Loader
# ----------------------------------------------------------

def resource_path(filename: str) -> str:
    """Return absolute path to a resource bundled with the app."""
    return os.path.join(os.path.dirname(__file__), filename)


@st.cache_data(show_spinner=True)
def load_contacts() -> pd.DataFrame:
    file_path = resource_path("event_data_for_app.parquet")

    if not os.path.exists(file_path):
        st.error(
            f"Could not find event_data_for_app.parquet at:\n{file_path}\n\n"
            "Make sure the file is in the same folder as app.py."
        )
        return pd.DataFrame()

    df = pd.read_parquet(file_path)
    df.columns = df.columns.str.lower().str.strip()
    return df.fillna("")


contacts = load_contacts()


# ----------------------------------------------------------
# Utility functions (DEFINED EARLY — REQUIRED)
# ----------------------------------------------------------

def similarity(a, b) -> float:
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def normalize_name(name):
    """Clean company name and strip out legal suffixes."""
    if not isinstance(name, str) or name.strip() == "":
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s()]", "", name)
    name = re.sub(
        r"\b(inc|llc|ltd|limited|corp|corporation|co|plc|sa|sao|sarl|bv|gmbh|ag|nv)\b",
        "",
        name,
    )
    name = re.sub(r"\s+", " ", name).strip()
    return name


def extract_abbreviation(text: str) -> str:
    """Extract abbreviation safely from parentheses."""
    if not isinstance(text, str):
        return ""
    matches = re.findall(r"\((.*?)\)", text)
    return matches[0].lower() if matches else ""


# ----------------------------------------------------------
# Precompute normalized fields & domain dictionary
# ----------------------------------------------------------

if "oracle account customer name" in contacts.columns:
    contacts["normalized_account"] = contacts["oracle account customer name"].apply(normalize_name)
    contacts["abbreviation"] = contacts["oracle account customer name"].apply(extract_abbreviation)

email_cols = [c for c in contacts.columns if "email" in c]

if email_cols:
    email_col = email_cols[0]
    domain_cols = [c for c in contacts.columns if "domain" in c]

    if domain_cols:
        domain_col = domain_cols[0]
    else:
        contacts["derived_domain"] = contacts[email_col].apply(
            lambda x: x.split("@")[-1].lower() if isinstance(x, str) and "@" in x else ""
        )
        domain_col = "derived_domain"

    domain_lookup = (
        contacts.sort_values(domain_col)
                .groupby(domain_col)
                .head(1)
                .set_index(domain_col)
                .to_dict("index")
    )
else:
    domain_lookup = {}


# ----------------------------------------------------------
# CONTACT MATCH FUNCTION
# ----------------------------------------------------------

def find_contact_matches(emails):
    results = []
    if contacts.empty:
        return pd.DataFrame()

    email_cols = [c for c in contacts.columns if "email" in c]
    domain_cols = [c for c in contacts.columns if "domain" in c]

    email_col = email_cols[0]

    if domain_cols:
        domain_col = domain_cols[0]
    else:
        contacts["derived_domain"] = contacts[email_col].apply(
            lambda x: x.split("@")[-1].lower()
            if isinstance(x, str) and "@" in x else ""
        )
        domain_col = "derived_domain"

    personal_fields = [
        "eloqua contacts first name",
        "eloqua contacts last name",
        "eloqua contacts job title",
        "eloqua contacts buying role",
        "eloqua contacts email address",
        "eloqua contacts do not email",
    ]

    output_cols = [c for c in contacts.columns if c not in ["match type", "match score"]]

    # ------------------------------------------------------
    # PROCESS INPUT EMAILS IN ORDER
    # ------------------------------------------------------
    for raw in emails:
        user_input = raw.strip()

        if user_input == "":
            row = {"input": "", "match type": "No Match", "match score": 0}
            for c in output_cols: row[c] = ""
            results.append(row)
            continue

        email = user_input.lower()

        # Exact match
        exact = contacts[contacts[email_col].str.lower() == email].copy()
        if not exact.empty:
            row = exact.iloc[0]
            out = {"input": user_input, "match type": "Exact Match", "match score": 100}
            for c in output_cols: out[c] = row.get(c, "")
            results.append(out)
            continue

        # Domain match
        domain = email.split("@")[-1]
        domain_match = contacts[contacts[domain_col].str.lower() == domain].copy()

        if not domain_match.empty:
            row = domain_match.iloc[0].copy()

            for col in personal_fields:
                if col in row: row[col] = ""

            out = {"input": user_input, "match type": "Domain Match", "match score": 90}
            for c in output_cols: out[c] = row.get(c, "")
            results.append(out)
            continue

        # No Match
        out = {"input": user_input, "match type": "No Match", "match score": 0}
        for c in output_cols: out[c] = ""
        results.append(out)

    return pd.DataFrame(results)


# ----------------------------------------------------------
# ACCOUNT MATCH FUNCTION — RAPIDFUZZ VERSION
# ----------------------------------------------------------

def find_account_matches(inputs):
    if contacts.empty:
        return pd.DataFrame()

    results = []
    acct_col = "oracle account customer name"

    # Ensure normalized + abbreviation fields exist
    contacts["normalized_account"] = contacts[acct_col].apply(normalize_name)
    contacts["abbreviation"] = contacts[acct_col].apply(extract_abbreviation)

    # Safe lists for RapidFuzz
    lookup_names = contacts["normalized_account"].fillna("").tolist()
    lookup_abbr  = contacts["abbreviation"].fillna("").tolist()

    output_cols = [
        "oracle account customer id",
        "oracle account customer name",
        "oracle account country",
        "oracle account business unit",
        "oracle account account segmentation",
        "ats team person name".
        "ae person name",
        "is partner",
        "oracle account line of business",
        "arr total arr",
        "arr next renewal date"
    ]

    # ------------------------------------------------------
    # PROCESS EACH INPUT
    # ------------------------------------------------------
    for raw in inputs:
        user_input = raw.strip()
        clean = user_input.lower()
        norm_input = normalize_name(clean)

        if not norm_input:
            out = {"input": user_input, "match type": "No Match", "match score": 0}
            for c in output_cols: out[c] = ""
            results.append(out)
            continue

        # 1. Abbreviation match
        if norm_input in lookup_abbr:
            idx = lookup_abbr.index(norm_input)
            row = contacts.iloc[idx]
            out = {"input": user_input, "match type": "Abbreviation Match", "match score": 95}
            for c in output_cols: out[c] = row.get(c, "")
            results.append(out)
            continue

        # 2. Exact normalized match
        if norm_input in lookup_names:
            idx = lookup_names.index(norm_input)
            row = contacts.iloc[idx]
            out = {"input": user_input, "match type": "Exact Match", "match score": 100}
            for c in output_cols: out[c] = row.get(c, "")
            results.append(out)
            continue

        # 3. RapidFuzz fuzzy match
        match = process.extractOne(norm_input, lookup_names, scorer=fuzz.ratio)

        if match:
            best_name, best_score, idx = match

            if best_score >= 85:
                row = contacts.iloc[idx]
                out = {
                    "input": user_input,
                    "match type": "Fuzzy Match",
                    "match score": round(best_score, 1),
                }
                for c in output_cols: out[c] = row.get(c, "")
                results.append(out)
                continue

        # 4. No match
        out = {"input": user_input, "match type": "No Match", "match score": 0}
        for c in output_cols: out[c] = ""
        results.append(out)

    return pd.DataFrame(results)


# ----------------------------------------------------------
# UI STYLING
# ----------------------------------------------------------

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #111827;
            color: #E5E7EB;
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        .citrix-title {{
            font-size: 1.9rem;
            font-weight: 600;
            color: {CITRIX_BLUE};
        }}
        .citrix-subtitle {{
            font-size: 0.95rem;
            color: #9CA3AF;
        }}
        .stTextArea textarea {{
            background-color: #1F2933;
            color: #F9FAFB;
            border-radius: 6px;
        }}
        .stTabs [data-baseweb="tab"] {{
            font-weight: 500;
            color: #D1D5DB;
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            color: {CITRIX_BLUE};
            border-bottom: 3px solid {CITRIX_BLUE};
        }}
        .stButton>button {{
            background-color: {CITRIX_BLUE};
            color: white;
            border-radius: 6px;
            border: none;
        }}
        .stButton>button:hover {{
            filter: brightness(1.08);
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------
# MAIN UI
# ----------------------------------------------------------

st.markdown('<div class="citrix-title">Citrix Event Lookup Tool</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="citrix-subtitle">'
    'Lookup event contacts and map account names to Citrix account records.'
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

if not contacts.empty:
    st.success(
        f"Loaded **{len(contacts):,}** rows • **{len(contacts.columns)}** columns "
        f"from `event_data_for_app.parquet`."
    )
else:
    st.error("No data loaded. Check your parquet file and restart the app.")

tab1, tab2 = st.tabs(["Contact Lookup", "Account Match"])


# -------------------------
# CONTACT TAB
# -------------------------
with tab1:
    st.subheader("Paste attendee emails (one per line):")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        emails_raw = st.text_area(
            "Emails",
            height=220,
            placeholder="one.email@example.com\nanother.person@company.com",
        )

    with col_right:
        st.markdown(
            """
            **How it works**  
            • Exact match  
            • Domain fallback  
            • Personal fields blanked  
            • Always one row per input  
            """
        )

    run_contacts = st.button("Run Contact Lookup", type="primary")

    if run_contacts:
        items = [x.strip() for x in emails_raw.splitlines()]
        with st.spinner("Searching contacts..."):
            df = find_contact_matches(items)

        if df.empty:
            st.warning("No matches found.")
        else:
            st.success(f"Generated **{len(df):,}** rows.")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📂 Download CSV",
                data=csv,
                file_name="contact_results.csv",
                mime="text/csv",
            )


# -------------------------
# ACCOUNT TAB
# -------------------------
with tab2:
    st.subheader("Paste account names (one per line):")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        accounts_raw = st.text_area(
            "Account Names",
            height=220,
            placeholder="Acme Corp\nAcme Corporation (ACME)\nContoso Ltd",
        )

    with col_right:
        st.markdown(
            """
            **Matching logic**  
            • Normalized exact match  
            • Abbreviation match  
            • RapidFuzz fuzzy match  
            • One row per input  
            • Input order preserved  
            """
        )

    run_accounts = st.button(" Run Account Lookup", type="primary")

    if run_accounts:
        items = [x.strip() for x in accounts_raw.splitlines()]
        with st.spinner("Matching accounts..."):
            df = find_account_matches(items)

        if df.empty:
            st.warning("No account matches found.")
        else:
            st.success(f"Generated **{len(df):,}** account rows.")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📂 Download CSV",
                data=csv,
                file_name="account_results.csv",
                mime="text/csv",
            )
