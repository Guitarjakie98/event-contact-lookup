import sys
import os
import numpy as np
import re
from difflib import SequenceMatcher
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import streamlit as st     # last import
import requests

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
            f"Could not find ContactDataApp2.1.parquett at:\n{file_path}\n\n"
            "Make sure the file is in the same folder as app.py."
        )
        return pd.DataFrame()

    df = pd.read_parquet(file_path)
    df.columns = df.columns.str.lower().str.strip()
    return df.fillna("")


contacts = load_contacts()

# ----------------------------------------------------------
# Load precomputed embeddings (fast startup)
# ----------------------------------------------------------
import numpy as np
import requests
import io
import streamlit as st

EMBEDDINGS_URL = "https://storage.googleapis.com/citrix-event-lookup/account_embeddings.npy"

@st.cache_resource(show_spinner=True)
def load_embeddings():
    # Remove the st.write!
    r = requests.get(EMBEDDINGS_URL)
    r.raise_for_status()
    return np.load(io.BytesIO(r.content))

# Load once at startup
with st.spinner("Loading embeddings…"):
    account_embeddings = load_embeddings()

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
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

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
    # Pre-load model + embeddings
    model = load_embedding_model()
    embeddings_all = account_embeddings
    norm_names = contacts["normalized_account"].tolist()

    # --- Pre-tokenize all accounts for Jaccard ---
    token_sets = [set(n.split()) for n in norm_names]

    # --- Normalize all user inputs at once ---
    norm_inputs = [normalize_name(x) for x in inputs]
    input_tokens = [set(n.split()) for n in norm_inputs]

    # --- Batch Encode Inputs Once (MAJOR SPEEDUP) ---
    input_embeddings = model.encode(
        norm_inputs,
        convert_to_tensor=True,
        batch_size=256,
        show_progress_bar=False
    )

    # --- Compute cosine similarity matrix (N x M) ---
    cosine_matrix = util.cos_sim(input_embeddings, embeddings_all)

    # --- Top-K semantic candidates for each input ---
    k = 15
    topk_scores, topk_idx = torch.topk(cosine_matrix, k=k, dim=1)
    
    # ---------------------------------------------------
    # PROCESS EACH INPUT
    # ---------------------------------------------------
    for i, raw in enumerate(inputs):
        user_input = raw.strip()
        norm_input = norm_inputs[i]
    
        # --- Empty input handling FIRST ---
        if not norm_input:
            out = {"input": user_input, "match type": "No Match", "match score": 0}
            for c in output_cols:
                out[c] = ""
            results.append(out)
            continue
    
        # --- Normalized Exact Match ---
        normalized_exact = contacts[contacts["normalized_account"] == norm_input]
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
    
        # --- Abbreviation Exact Match ---
        if "abbreviation" in contacts.columns and norm_input in contacts["abbreviation"].values:
            row = contacts[contacts["abbreviation"] == norm_input].iloc[0]
            out = {
                "input": user_input,
                "match type": "Abbreviation Match",
                "match score": 100.0,
            }
            for display_col, actual_col in output_cols.items():
                out[display_col] = row.get(actual_col, "")
            results.append(out)
            continue
    
        # --- Candidate pool (only top-K rows) ---
        candidate_ids = topk_idx[i].cpu().numpy()
        candidate_names = [norm_names[j] for j in candidate_ids]
    
        # --- Fuzzy scores ---
        fuzzy = np.array([fuzz.ratio(norm_input, c) / 100 for c in candidate_names])
    
        # --- Jaccard scores ---
        jac = np.array([
            len(input_tokens[i] & token_sets[j]) /
            len(input_tokens[i] | token_sets[j]) if token_sets[j] else 0
            for j in candidate_ids
        ])
    
        # --- Combine scores ---
        hybrid = (
            0.65 * topk_scores[i].cpu().numpy() +
            0.25 * fuzzy +
            0.10 * jac
        )
    
        best_local_index = int(np.argmax(hybrid))
        best_score = float(hybrid[best_local_index])
        best_account_index = candidate_ids[best_local_index]
    
        row = contacts.iloc[best_account_index]
        row = row.drop(labels=["party_number"], errors="ignore")
        final_score_percent = round(best_score * 100, 1)
    
        # --- Match categories ---
        if best_score >= 0.70:
            match_type = "High Confidence Match"
        elif best_score >= 0.55:
            match_type = "Low Confidence Match"
        else:
            match_type = "No Match"

        out = {
            "input": user_input,
            "match type": match_type,
            "match score": final_score_percent,
        }

        for display_col, actual_col in output_cols.items():
            out[display_col] = row.get(actual_col, "") if match_type != "No Match" else ""

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
