import os
import numpy as np
import re
from rapidfuzz import fuzz
import pandas as pd
import streamlit as st


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
    file_path = resource_path("ContactDataApp2.1.parquet")

    if not os.path.exists(file_path):
        st.error(
            f"Could not find ContactDataApp2.1.parquet at:\n{file_path}\n\n"
            "Make sure the file is in the same folder as app.py."
        )
        return pd.DataFrame()

    df = pd.read_parquet(file_path)
    df.columns = df.columns.str.lower().str.strip()
    return df.fillna("")


contacts = load_contacts()

# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------

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

# Create required columns BEFORE embeddings
contacts["normalized_account"] = contacts["customer_name"].apply(normalize_name)
contacts["abbreviation"] = contacts["customer_name"].apply(extract_abbreviation)

# Email domain logic should also be here
if "email_address" in contacts.columns:
    contacts["email_domain"] = contacts["email_address"].apply(
        lambda x: x.split("@")[-1].lower() if isinstance(x, str) and "@" in x else ""
    )
else:
    contacts["email_domain"] = ""

# ----------------------------------------------------------
# Embedding Model Loader (cached)
# ----------------------------------------------------------

def find_contact_matches(emails):
    results = []
    if contacts.empty:
        return pd.DataFrame()

    # Identify the email column explicitly
    email_cols = [c for c in contacts.columns if c == "email_address"]
    if not email_cols:
        return pd.DataFrame([{
            "input": e, "match type": "No Match", "match score": 0
        } for e in emails])

    email_col = email_cols[0]

    # Determine domain column
    if "email_domain" in contacts.columns:
        domain_col = "email_domain"
    else:
        contacts["email_domain"] = contacts[email_col].apply(
            lambda x: x.split("@")[-1].lower() if isinstance(x,str) and "@" in x else ""
        )
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
        if c not in ["match type", "match score", "party_number", "normalized_account", "abbreviation" ]
    ]

    # ------------------------------------------------------
    # PROCESS INPUT EMAILS IN ORDER
    # ------------------------------------------------------
    for raw in emails:
        user_input = raw.strip()

        # Empty input
        if user_input == "":
            row = {"input": "", "match type": "No Match", "match score": 0}
            for c in output_cols:
                row[c] = ""
            results.append(row)
            continue

        email = user_input.lower()

        # Exact match
        exact = contacts[contacts[email_col].str.lower() == email]
        if not exact.empty:
            row = exact.iloc[0]
            out = {"input": user_input, "match type": "Exact Match", "match score": 100}
            for c in output_cols:
                out[c] = row.get(c, "")
            results.append(out)
            continue

        # Domain match
        domain = email.split("@")[-1]
        domain_match = contacts[contacts[domain_col].str.lower() == domain]

        if not domain_match.empty:
            row = domain_match.iloc[0].copy()

            # Blank personal fields so we don’t leak wrong contact data
            for col in personal_fields:
                if col in row:
                    row[col] = ""

            out = {"input": user_input, "match type": "Domain Match", "match score": 90}
            for c in output_cols:
                out[c] = row.get(c, "")
            results.append(out)
            continue

        # No match
        out = {"input": user_input, "match type": "No Match", "match score": 0}
        for c in output_cols:
            out[c] = ""
        results.append(out)

    # ----------------------------------------------------------
    # END OF FUNCTION — Format output dataframe (AFTER LOOP)
    # ----------------------------------------------------------
    df = pd.DataFrame(results)

    # Column order
    desired_order = [
        "input",
        "match type",
        "match score",
        "customer_name",
        "customer_id",
        "first_name",
        "last_name",
        "job_title",
        "email_address",
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

    desired_order = [c for c in desired_order if c in df.columns]

    df = df.reindex(columns = desired_order + [c for c in df.columns if c not in desired_order])

    return df
     
# ----------------------------------------------------------
# ACCOUNT MATCH FUNCTION — RAPIDFUZZ VERSION
# ----------------------------------------------------------

def find_account_matches(inputs):
    if contacts.empty:
        return pd.DataFrame()

    results = []

    output_cols = {
        "customer_name": "customer_name",
        "customer_id": "customer_id",
        "account_segmentation": "account_segmentation",
        "country": "country",
        "line_of_business": "line_of_business",
        "level14_territory_name": "level14_territory_name",
        "arr": "arr",
        "ae_name": "ae_name",
        "ats_name": "ats_name",
        "ispartner": "ispartner",
        "account_engagement_score": "account_engagement_score",
        "next_renewal_date": "next_renewal_date",
    }

    # Get unique accounts for matching
    unique_accounts = contacts.drop_duplicates(subset="customer_id", keep="first")

    for raw in inputs:
        user_input = raw.strip()
        norm_input = normalize_name(user_input)

        # Empty input
        if not norm_input:
            out = {"input": user_input, "match type": "No Match", "match score": 0}
            for c in output_cols:
                out[c] = ""
            results.append(out)
            continue

        # Normalized Exact Match
        normalized_exact = unique_accounts[unique_accounts["normalized_account"] == norm_input]
        if not normalized_exact.empty:
            row = normalized_exact.iloc[0]
            out = {
                "input": user_input,
                "match type": "Normalized Exact Match",
                "match score": 100.0,
            }
            for display_col, actual_col in output_cols.items():
                out[display_col] = row.get(actual_col, "")
            results.append(out)
            continue

        # Abbreviation Exact Match
        if "abbreviation" in unique_accounts.columns:
            abbr_match = unique_accounts[unique_accounts["abbreviation"] == norm_input]
            if not abbr_match.empty:
                row = abbr_match.iloc[0]
                out = {
                    "input": user_input,
                    "match type": "Abbreviation Match",
                    "match score": 100.0,
                }
                for display_col, actual_col in output_cols.items():
                    out[display_col] = row.get(actual_col, "")
                results.append(out)
                continue

        # Fuzzy matching
        best_score = 0
        best_row = None
        
        for idx, row in unique_accounts.iterrows():
            score = fuzz.ratio(norm_input, row["normalized_account"])
            if score > best_score:
                best_score = score
                best_row = row

        # Determine match type - only show matches with 90% or higher confidence
        if best_score >= 90:
            match_type = "High Confidence Match"
        else:
            match_type = "No Match"

        out = {
            "input": user_input,
            "match type": match_type,
            "match score": float(best_score),
        }

        for display_col, actual_col in output_cols.items():
            out[display_col] = (
                best_row.get(actual_col, "") if match_type != "No Match" and best_row is not None else ""
            )

        results.append(out)

    return pd.DataFrame(results)

# ----------------------------------------------------------
# TITLE TO ACCOUNT FUNCTION
# ----------------------------------------------------------

def find_title_to_account_matches(inputs):
    """Match job titles to contacts within specific accounts.
    
    Args:
        inputs: List of tuples (account_name, job_title)
    
    Returns:
        DataFrame with matching contacts (same format as contact lookup)
    """
    if contacts.empty:
        return pd.DataFrame()

    results = []
    
    # Use same output format as contact lookup (Tab 1)
    output_cols = [
        c for c in contacts.columns
        if c not in ["match type", "match score", "party_number", "normalized_account", "abbreviation"]
    ]

    # Get unique accounts for matching
    unique_accounts = contacts.drop_duplicates(subset="customer_id", keep="first")

    for account_input, job_title_input in inputs:
        account_name = account_input.strip()
        job_title = job_title_input.strip()
        norm_account = normalize_name(account_name)

        # Empty inputs
        if not job_title or not norm_account:
            out = {
                "input_account": account_name,
                "input_job_title": job_title,
                "match type": "No Match",
                "match score": 0,
            }
            for c in output_cols:
                out[c] = ""
            results.append(out)
            continue

        # Find matching account first - use same logic as Account Match tab
        matched_account_id = None
        account_match_score = 0

        # Normalized Exact Match
        normalized_exact = unique_accounts[unique_accounts["normalized_account"] == norm_account]
        if not normalized_exact.empty:
            matched_account_id = normalized_exact.iloc[0]["customer_id"]
            account_match_score = 100
        
        # Abbreviation Exact Match
        elif "abbreviation" in unique_accounts.columns:
            abbr_match = unique_accounts[unique_accounts["abbreviation"] == norm_account]
            if not abbr_match.empty:
                matched_account_id = abbr_match.iloc[0]["customer_id"]
                account_match_score = 100
        
        # Fuzzy matching (≥90% like Account Match tab)
        if matched_account_id is None:
            best_score = 0
            best_row = None
            
            for idx, row in unique_accounts.iterrows():
                score = fuzz.ratio(norm_account, row["normalized_account"])
                if score > best_score:
                    best_score = score
                    best_row = row

            if best_score >= 90:
                matched_account_id = best_row["customer_id"]
                account_match_score = best_score

        # If no account match ≥90%, return no match
        if matched_account_id is None:
            out = {
                "input_account": account_name,
                "input_job_title": job_title,
                "match type": "No Account Match",
                "match score": 0,
            }
            for c in output_cols:
                out[c] = ""
            results.append(out)
            continue

        # Find all contacts at this account and score their job titles
        account_contacts = contacts[contacts["customer_id"] == matched_account_id]
        
        if "job_title" in account_contacts.columns:
            job_title_lower = job_title.lower()
            scored_contacts = []
            
            # Score all contacts with job titles
            for _, row in account_contacts.iterrows():
                contact_title = str(row.get("job_title", "")).lower()
                if contact_title:
                    title_score = fuzz.ratio(job_title_lower, contact_title)
                    scored_contacts.append((title_score, row))
            
            if scored_contacts:
                # Sort by score descending (best matches first)
                scored_contacts.sort(key=lambda x: x[0], reverse=True)
                
                # Return all matches, sorted by score
                for title_score, row in scored_contacts:
                    out = {
                        "input_account": account_name,
                        "input_job_title": job_title,
                        "match type": "Title + Account Match",
                        "match score": title_score,
                    }
                    for c in output_cols:
                        out[c] = row.get(c, "")
                    results.append(out)
            else:
                # Account found but no job titles
                out = {
                    "input_account": account_name,
                    "input_job_title": job_title,
                    "match type": "No Title Match",
                    "match score": 0,
                }
                for c in output_cols:
                    out[c] = ""
                results.append(out)
        else:
            # No job_title column
            out = {
                "input_account": account_name,
                "input_job_title": job_title,
                "match type": "No Job Title Column",
                "match score": 0,
            }
            for c in output_cols:
                out[c] = ""
            results.append(out)

    # Format output dataframe with same column order as Tab 1
    df = pd.DataFrame(results)
    
    desired_order = [
        "input_account",
        "input_job_title",
        "match type",
        "match score",
        "customer_name",
        "customer_id",
        "first_name",
        "last_name",
        "job_title",
        "email_address",
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
    
    desired_order = [c for c in desired_order if c in df.columns]
    df = df.reindex(columns=desired_order + [c for c in df.columns if c not in desired_order])
    
    return df

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

tab1, tab2, tab3 = st.tabs(["Contact Lookup", "Account Match", "Title to Account"])


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
            • Embedding semnatic match  
            • One row per input  
            • Input order preserved  
            """
        )

    run_accounts = st.button("Run Account Lookup", type="primary")

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


# -------------------------
# TITLE TO ACCOUNT TAB
# -------------------------
with tab3:
    st.subheader("Paste account names and job titles (tab or comma separated):")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        title_account_raw = st.text_area(
            "Account Name and Job Title (2 columns)",
            height=220,
            placeholder="American Express\tManager - Data Science\nAmerican Express\tEnterprise Strategy, Manager\nAmerican Express\tAI Product Manager",
        )

    with col_right:
        st.markdown(
            """
            **How it works**  
            • Paste 2 columns (tab/comma separated)
            • Column 1: Account name  
            • Column 2: Job title  
            • Account match ≥90%
            • Returns ALL contacts at that account
            • Sorted by job title similarity (best first)
            """
        )

    run_title_account = st.button("Run Title to Account Lookup", type="primary", key="run_title_account")

    if run_title_account:
        lines = [x.strip() for x in title_account_raw.splitlines() if x.strip()]
        items = []
        
        for line in lines:
            # Try tab separator first, then comma
            if "\t" in line:
                parts = line.split("\t", 1)
            elif "," in line:
                parts = line.split(",", 1)
            else:
                # Single column, skip
                continue
            
            if len(parts) == 2:
                items.append((parts[0].strip(), parts[1].strip()))
        
        if not items:
            st.warning("No valid input. Please paste 2 columns separated by tab or comma.")
        else:
            with st.spinner("Searching for contacts..."):
                df = find_title_to_account_matches(items)

            if df.empty:
                st.warning("No matches found.")
            else:
                st.success(f"Generated **{len(df):,}** rows.")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📂 Download CSV",
                    data=csv,
                    file_name="title_account_results.csv",
                    mime="text/csv",
                )

