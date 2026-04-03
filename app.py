import os
import pandas as pd
import streamlit as st

from config import PARQUET_PATH
from matching import prepare_contacts, find_contact_matches, find_account_matches, find_title_to_account_matches

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
    if not os.path.exists(PARQUET_PATH):
        st.error(
            f"Could not find ContactDataApp2.1.parquet at:\n{PARQUET_PATH}\n\n"
            "Set the DATA_DIR environment variable or place the file in master_data/."
        )
        return pd.DataFrame()

    return prepare_contacts(pd.read_parquet(PARQUET_PATH))


contacts = load_contacts()


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
        f"from `ContactDataApp2.1.parquet`."
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
            df = find_contact_matches(contacts, items)

        if df.empty:
            st.warning("No matches found.")
        else:
            st.success(f"Generated **{len(df):,}** rows.")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
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
            • Fuzzy match (>=90%)
            • One row per input
            • Input order preserved
            """
        )

    run_accounts = st.button("Run Account Lookup", type="primary")

    if run_accounts:
        items = [x.strip() for x in accounts_raw.splitlines()]
        with st.spinner("Matching accounts..."):
            df = find_account_matches(contacts, items)

        if df.empty:
            st.warning("No account matches found.")
        else:
            st.success(f"Generated **{len(df):,}** account rows.")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
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
            • Account match >=90%
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
                continue

            if len(parts) == 2:
                items.append((parts[0].strip(), parts[1].strip()))

        if not items:
            st.warning("No valid input. Please paste 2 columns separated by tab or comma.")
        else:
            with st.spinner("Searching for contacts..."):
                df = find_title_to_account_matches(contacts, items)

            if df.empty:
                st.warning("No matches found.")
            else:
                st.success(f"Generated **{len(df):,}** rows.")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name="title_account_results.csv",
                    mime="text/csv",
                )
