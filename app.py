import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz
import requests

# ----------------------------------------------------------
# STREAMLIT CONFIG
# ----------------------------------------------------------

st.set_page_config(
    page_title="Citrix Event Lookup Tool",
    layout="wide"
)

st.set_option("server.fileWatcherType", "none")

CITRIX_BLUE = "#009FD9"

# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------

def resource_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), filename)

def normalize_name(name):
    if not isinstance(name, str) or name.strip() == "":
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s()]", "", name)
    name = re.sub(
        r"\b(inc|llc|ltd|limited|corp|corporation|co|plc|sa|sarl|bv|gmbh|ag|nv)\b",
        "",
        name,
    )
    name = re.sub(r"\s+", " ", name).strip()
    return name

def extract_abbreviation(text):
    if not isinstance(text, str):
        return ""
    matches = re.findall(r"\((.*?)\)", text)
    return matches[0].lower() if matches else ""

# ----------------------------------------------------------
# LOAD CONTACT DATA
# ----------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_contacts():
    path = resource_path("ContactDataApp2.1.parquet")

    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df.columns = df.columns.str.lower().str.strip()
    df = df.fillna("")
    return df

contacts = load_contacts()

# Create derived columns
contacts["normalized_account"] = contacts["customer_name"].apply(normalize_name)
contacts["abbreviation"] = contacts["customer_name"].apply(extract_abbreviation)

if "email_address" in contacts.columns:
    contacts["email_domain"] = contacts["email_address"].apply(
        lambda x: x.split("@")[-1].lower() if "@" in x else ""
    )
else:
    contacts["email_domain"] = ""

# ----------------------------------------------------------
# CONTACT MATCH (EMAIL)
# ----------------------------------------------------------

def find_contact_matches(emails):
    results = []

    if contacts.empty:
        return pd.DataFrame()

    email_col = "email_address"
    domain_col = "email_domain"

    personal_fields = [
        "first_name",
        "last_name",
        "job_title",
        "sales_buying_role_code",
        "email_address",
        "do_not_email_flag",
    ]

    output_cols = [
        c for c in contacts.columns
        if c not in ["normalized_account", "abbreviation"]
    ]

    for raw in emails:
        user_input = raw.strip().lower()

        if user_input == "":
            results.append({"input": "", "match type": "No Match", "match score": 0})
            continue

        exact = contacts[contacts[email_col].str.lower() == user_input]
        if not exact.empty:
            row = exact.iloc[0]
            out = {"input": raw, "match type": "Exact Match", "match score": 100}
            for c in output_cols:
                out[c] = row.get(c, "")
            results.append(out)
            continue

        domain = user_input.split("@")[-1]
        domain_match = contacts[contacts[domain_col] == domain]

        if not domain_match.empty:
            row = domain_match.iloc[0].copy()
            for col in personal_fields:
                if col in row:
                    row[col] = ""

            out = {"input": raw, "match type": "Domain Match", "match score": 90}
            for c in output_cols:
                out[c] = row.get(c, "")
            results.append(out)
            continue

        out = {"input": raw, "match type": "No Match", "match score": 0}
        for c in output_cols:
            out[c] = ""
        results.append(out)

    return pd.DataFrame(results)

# ----------------------------------------------------------
# ACCOUNT MATCH (FUZZY + JACCARD)
# ----------------------------------------------------------

def find_account_matches(inputs):

    if contacts.empty:
        return pd.DataFrame()

    results = []

    norm_names = contacts["normalized_account"].tolist()
    token_sets = [set(n.split()) for n in norm_names]

    output_cols = [
        "customer_name",
        "customer_id",
        "account_segmentation",
        "country",
        "line_of_business",
        "level14_territory_name",
        "arr",
        "ae_name",
        "ats_name",
        "ispartner",
        "account_engagement_score",
        "next_renewal_date",
    ]

    for raw in inputs:
        user_input = raw.strip()
        norm_input = normalize_name(user_input)

        if norm_input == "":
            results.append({"input": user_input, "match type": "No Match", "match score": 0})
            continue

        # Normalized exact
        exact = contacts[contacts["normalized_account"] == norm_input]
        if not exact.empty:
            row = exact.iloc[0]
            out = {"input": user_input, "match type": "Normalized Exact Match", "match score": 100}
            for c in output_cols:
                out[c] = row.get(c, "")
            results.append(out)
            continue

        # Abbreviation
        if norm_input in contacts["abbreviation"].values:
            row = contacts[contacts["abbreviation"] == norm_input].iloc[0]
            out = {"input": user_input, "match type": "Abbreviation Match", "match score": 100}
            for c in output_cols:
                out[c] = row.get(c, "")
            results.append(out)
            continue

        input_tokens = set(norm_input.split())

        fuzzy_scores = np.array([
            fuzz.ratio(norm_input, c) / 100 for c in norm_names
        ])

        jac_scores = np.array([
            len(input_tokens & t) / len(input_tokens | t) if t else 0
            for t in token_sets
        ])

        hybrid = 0.7 * fuzzy_scores + 0.3 * jac_scores

        best_idx = int(np.argmax(hybrid))
        best_score = hybrid[best_idx]
        row = contacts.iloc[best_idx]

        if best_score >= 0.70:
            match_type = "High Confidence Match"
        elif best_score >= 0.55:
            match_type = "Low Confidence Match"
        else:
            match_type = "No Match"

        out = {
            "input": user_input,
            "match type": match_type,
            "match score": round(best_score * 100, 1),
        }

        for c in output_cols:
            out[c] = row.get(c, "") if match_type != "No Match" else ""

        results.append(out)

    return pd.DataFrame(results)

# ----------------------------------------------------------
# UI STYLES
# ----------------------------------------------------------

st.markdown(
f"""
<style>
.stApp {{
    background-color:#111827;
    color:#E5E7EB;
}}
.stButton>button {{
    background-color:{CITRIX_BLUE};
    color:white;
}}
</style>
""",
unsafe_allow_html=True
)

# ----------------------------------------------------------
# MAIN UI
# ----------------------------------------------------------

st.markdown("<h2 style='color:#009FD9'>Citrix Event Lookup Tool</h2>", unsafe_allow_html=True)

if contacts.empty:
    st.error("No data loaded.")
else:
    st.success(f"Loaded {len(contacts):,} rows")

tab1, tab2 = st.tabs(["Contact Lookup", "Account Match"])

# ---------------- Contact Tab ----------------

with tab1:
    emails_raw = st.text_area("Paste attendee emails (one per line)", height=220)
    if st.button("Run Contact Lookup"):
        items = [x.strip() for x in emails_raw.splitlines()]
        df = find_contact_matches(items)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "contact_results.csv")

# ---------------- Account Tab ----------------

with tab2:
    accounts_raw = st.text_area("Paste account names (one per line)", height=220)
    if st.button("Run Account Lookup"):
        items = [x.strip() for x in accounts_raw.splitlines()]
        df = find_account_matches(items)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "account_results.csv")

