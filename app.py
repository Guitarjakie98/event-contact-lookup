import sys
import os
import re
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st     # last import

# MUST be the first Streamlit command
st.set_page_config(
    page_title="Citrix Event Lookup Tool",
    layout="wide",
    page_icon="📊",
)

CITRIX_BLUE = "#009FD9"   # constants AFTER set_page_config are fine

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

    contacts = pd.read_parquet(file_path)
    contacts.columns = contacts.columns.str.lower().str.strip()
    return contacts.fillna("")


contacts = load_contacts()

# ----------------------------------------------------------
# Utility
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
    """Extract abbreviation safely from parentheses, if present."""
    if not isinstance(text, str):
        return ""
    matches = re.findall(r"\((.*?)\)", text)
    return matches[0].lower() if matches else ""


# ----------------------------------------------------------
# CONTACT LOOKUP
# ----------------------------------------------------------

def find_contact_matches(emails):
    if contacts.empty:
        return pd.DataFrame()

    results, missing_records = [], []

    # Identify candidate columns
    email_col_candidates = [c for c in contacts.columns if "email" in c]
    domain_col_candidates = [c for c in contacts.columns if "domain" in c]

    email_col = email_col_candidates[0] if email_col_candidates else None
    domain_col = domain_col_candidates[0] if domain_col_candidates else None

    if not email_col:
        st.error("No email column found in the dataset.")
        return pd.DataFrame()

    # If no domain col exists, derive it from email field
    if not domain_col:
        contacts["derived_domain"] = contacts[email_col].apply(
            lambda x: x.split("@")[-1].lower()
            if isinstance(x, str) and "@" in x
            else ""
        )
        domain_col = "derived_domain"

    # ------------------------------------------------------
    # PROCESS EACH INPUT EMAIL
    # ------------------------------------------------------
    for email in emails:
        email = email.strip().lower()
        if not email:
            continue

        # ----- Exact match -----
        exact = contacts[contacts[email_col].str.lower() == email].copy()
        if not exact.empty:
            exact.insert(0, "match type", "Exact Match")
            exact.insert(1, "match score", 100)
            results.append(exact)
            continue

        # ----- Domain match -----
        domain = email.split("@")[-1]
        domain_match = contacts[contacts[domain_col].str.lower() == domain].copy()

        if not domain_match.empty:

            # Keep only ONE ROW instead of hundreds
            single = domain_match.iloc[[0]].copy()

            # Clear personal contact fields
            personal_fields = [
                "eloqua contacts first name",
                "eloqua contacts last name",
                "eloqua contacts job title",
                "eloqua contacts buying role",
                "eloqua contacts email address",
                "eloqua contacts do not email",
            ]
            for col in personal_fields:
                if col in single.columns:
                    single[col] = ""

            # Insert match metadata
            single.insert(0, "match type", "Domain Match")
            single.insert(1, "match score", 90)

            results.append(single)

        else:
            missing_records.append({
                "match type": "No Match",
                "match score": 0,
                "input": email
            })

    # ------------------------------------------------------
    # Combine all results
    # ------------------------------------------------------
    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    if missing_records:
        combined = pd.concat(
            [combined, pd.DataFrame(missing_records)],
            ignore_index=True,
        )

    return combined

# ----------------------------------------------------------
# ACCOUNT MATCH
# ----------------------------------------------------------

def find_account_matches(inputs):
    """Find matching accounts by name or abbreviation (with fuzzy threshold)."""
    if contacts.empty:
        return pd.DataFrame()

    results = []
    account_col = "oracle account customer name"

    if account_col not in contacts.columns:
        st.error(
            "Missing 'oracle account customer name' in data. "
            "Check that your parquet file has this column."
        )
        return pd.DataFrame()

    # Prepare normalized + abbreviation columns (cached on the df object)
    if "normalized_account" not in contacts.columns:
        contacts["normalized_account"] = contacts[account_col].apply(normalize_name)
    if "abbreviation" not in contacts.columns:
        contacts["abbreviation"] = contacts[account_col].apply(extract_abbreviation)

    for raw in inputs:
        user_input = raw.strip().lower()
        norm_input = normalize_name(user_input)
        if not norm_input:
            continue

        best_row, best_score, match_type = None, 0.0, "No Match"

        for _, row in contacts.iterrows():
            acct_norm = row["normalized_account"]
            abbrev = row["abbreviation"]

            if norm_input == acct_norm:
                score, match_type = 1.0, "Exact Match"

            elif abbrev and abbrev == norm_input:
                score, match_type = 0.95, "Strong Fuzzy Match"

            else:
                score = similarity(norm_input, acct_norm)
                if score >= 0.90:
                    match_type = "Fuzzy Match"
                elif score >= 0.90:  # kept as-is from your original logic
                    match_type = "Weak Fuzzy Match"
                else:
                    continue

            if score > best_score:
                best_row = row
                best_score = score

        if best_row is not None and best_score >= 0.85:
            if best_score == 1.0:
                match_type = "Exact Match"

            match_entry = {
                "input": raw,
                "match type": match_type,
                "match score": round(best_score * 100, 1),
                "oracle account customer id": best_row.get(
                    "oracle account customer id", ""
                ),
                "oracle account customer name": best_row.get(
                    "oracle account customer name", ""
                ),
                "oracle account country": best_row.get(
                    "oracle account country", ""
                ),
                "oracle account business unit": best_row.get(
                    "oracle account business unit", ""
                ),
                "oracle account segmentation": best_row.get(
                    "oracle account segmentation", ""
                ),
                "is partner": best_row.get("is partner", ""),
                "oracle account line of business": best_row.get(
                    "oracle account line of business", ""
                ),
                "arr total arr": best_row.get("arr total arr", ""),
                "arr next renewal date": best_row.get("arr next renewal date", ""),
            }
            results.append(match_entry)

    if not results:
        return pd.DataFrame()

    results_df = (
        pd.DataFrame(results)
        .sort_values(by="match score", ascending=False)
        .reset_index(drop=True)
    )

    print(f"✔ Account match complete — {len(results_df)} matches found.")
    return results_df


# ----------------------------------------------------------
# UI STYLING
# ----------------------------------------------------------

# Custom CSS
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
# MAIN LAYOUT
# ----------------------------------------------------------

st.markdown('<div class="citrix-title">Citrix Event Lookup Tool</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="citrix-subtitle">'
    'Lookup event contacts and map account names to Citrix account records.'
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Dataset status
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
            • Exact match on email address  
            • Fallback to domain-only match  
            • Personal fields are blanked on domain matches  
            """
        )

    run_contacts = st.button("🔍 Run Contact Lookup", type="primary")

    if run_contacts:
        items = [x.strip() for x in emails_raw.splitlines() if x.strip()]
        if not items:
            st.warning("Please enter at least one email.")
        else:
            with st.spinner("Searching contacts..."):
                df = find_contact_matches(items)

            if df.empty:
                st.warning("No matches found (≥ 0% threshold).")
            else:
                st.success(f"Found **{len(df):,}** rows.")
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
            • Normalizes names (removes Inc, LLC, etc.)  
            • Uses abbreviations in parentheses (e.g., *Advanced Micro Devices (AMD)*)  
            • Fuzzy string similarity with thresholds  
            """
        )

    run_accounts = st.button("🏢 Run Account Lookup", type="primary")

    if run_accounts:
        items = [x.strip() for x in accounts_raw.splitlines() if x.strip()]
        if not items:
            st.warning("Please enter at least one account name.")
        else:
            with st.spinner("Matching accounts..."):
                df = find_account_matches(items)

            if df.empty:
                st.warning("No account matches found above the threshold.")
            else:
                st.success(f"Found **{len(df):,}** account matches.")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📂 Download CSV",
                    data=csv,
                    file_name="account_results.csv",
                    mime="text/csv",
                )
