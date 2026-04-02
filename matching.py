# matching.py
#
# Shared matching logic for contact lookup, account matching, and
# title-to-account matching.  No Streamlit, no FastAPI — pure functions
# that operate on DataFrames.

import re
from typing import List, Tuple

import pandas as pd
from rapidfuzz import fuzz, process as rfuzz_process

# ============================================================
# Personal / generic email domains to skip
# ============================================================

GENERIC_DOMAINS = {
    "gmail.com", "googlemail.com",
    "hotmail.com", "hotmail.co.uk", "hotmail.fr", "hotmail.de", "hotmail.es",
    "hotmail.it", "hotmail.ca", "hotmail.com.br", "hotmail.com.ar",
    "live.com", "live.co.uk", "live.fr", "live.de", "live.ca",
    "msn.com", "outlook.com", "outlook.com.br",
    "yahoo.com", "yahoo.co.uk", "yahoo.co.in", "yahoo.fr", "yahoo.de",
    "yahoo.es", "yahoo.it", "yahoo.ca", "yahoo.com.br", "yahoo.com.ar",
    "yahoo.com.au", "yahoo.com.mx", "ymail.com",
    "aol.com",
    "icloud.com", "me.com", "mac.com",
    "protonmail.com", "proton.me",
    "mail.com", "email.com",
    "zoho.com",
    "gmx.com", "gmx.net", "gmx.de",
    "web.de", "t-online.de",
    "laposte.net", "orange.fr", "wanadoo.fr", "free.fr", "sfr.fr",
    "libero.it",
    "yandex.com", "yandex.ru",
    "qq.com", "163.com", "126.com",
    "rediffmail.com",
    "terra.com.br", "uol.com.br", "bol.com.br",
    "naver.com",
    "comcast.net", "verizon.net", "att.net", "sbcglobal.net",
    "cox.net", "charter.net", "earthlink.net",
    "shaw.ca", "rogers.com", "sympatico.ca",
    "bigpond.com", "bigpond.net.au", "optusnet.com.au",
    "btinternet.com", "virginmedia.com", "sky.com",
    "tiscali.it", "alice.it",
    "seznam.cz",
    "wp.pl", "onet.pl", "interia.pl",
    "inbox.com",
    "fastmail.com", "fastmail.fm",
    "tutanota.com", "hushmail.com",
    "mail.ru",
}

# ============================================================
# Utilities
# ============================================================

def normalize_name(name):
    """Clean company name: lowercase, strip punctuation, remove legal suffixes."""
    if not isinstance(name, str) or name.strip() == "":
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s()]", "", name)
    name = re.sub(
        r"\b(inc|llc|ltd|limited|corp|corporation|co|plc|sa|sao|sarl|bv|gmbh|ag|nv)\b",
        "",
        name,
    )
    return re.sub(r"\s+", " ", name).strip()


def extract_abbreviation(text: str) -> str:
    """Extract abbreviation from parentheses, e.g. 'JACOBS(KLING STUBBINS)' -> 'kling stubbins'."""
    if not isinstance(text, str):
        return ""
    matches = re.findall(r"\((.*?)\)", text)
    return matches[0].lower() if matches else ""


# ============================================================
# Data preparation
# ============================================================

def prepare_contacts(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and add derived columns.

    Accepts a raw parquet DataFrame (either snake_case or Eloqua-format columns)
    and returns a cleaned DataFrame with normalized_account, abbreviation, and
    email_domain columns.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Rename Eloqua-format columns if present
    rename_map = {
        "oracle account customer name": "customer_name",
        "oracle account customer id": "customer_id",
        "eloqua contacts email address": "email_address",
        "eloqua contacts email address domain": "email_domain",
        "eloqua contacts first name": "first_name",
        "eloqua contacts last name": "last_name",
        "eloqua contacts job title": "job_title",
        "oracle account account segmentation": "account_segmentation",
        "oracle account country": "country",
        "oracle account line of business": "line_of_business",
        "ae person name": "ae_name",
        "ae level14 territory name": "level15_territory_name",
        "ats team person name": "ats_name",
        "arr total arr": "arr",
        "arr next renewal date": "next_renewal_date",
        "is partner": "ispartner",
        "eloqua accounts account engagement score": "account_engagement_score",
    }
    df = df.rename(columns=rename_map)
    df = df.fillna("")

    # Derived columns
    df["normalized_account"] = df["customer_name"].apply(normalize_name)
    df["abbreviation"] = df["customer_name"].apply(extract_abbreviation)

    if "email_address" in df.columns:
        df["email_domain"] = df["email_address"].apply(
            lambda x: x.split("@")[-1].lower() if isinstance(x, str) and "@" in x else ""
        )

    return df


# ============================================================
# Matching logic
# ============================================================

# Columns that contain personal contact info — blanked on domain-only matches
_PERSONAL_FIELDS = [
    "first_name",
    "last_name",
    "job_title",
    "sales_buying_role_code",
    "email_address",
    "do_not_email_flag",
]

# Standard output column order
_CONTACT_COLUMN_ORDER = [
    "input",
    "match type",
    "match score",
    "customer_name",
    "customer_id",
    "first_name",
    "last_name",
    "job_title",
    "email_address",
    "email_domain",
    "do_not_email",
    "do_not_call",
    "account_segmentation",
    "account_status",
    "business_unit",
    "country",
    "prime_geo",
    "overlay_geo",
    "level15_territory_name",
    "arr",
    "ae_name",
    "ats_name",
    "ats_email",
    "ispartner",
    "next_renewal_date",
]

_ACCOUNT_OUTPUT_COLS = {
    "customer_name": "customer_name",
    "customer_id": "customer_id",
    "account_segmentation": "account_segmentation",
    "country": "country",
    "prime_geo": "prime_geo",
    "overlay_geo": "overlay_geo",
    "level15_territory_name": "level15_territory_name",
    "arr": "arr",
    "ae_name": "ae_name",
    "ats_name": "ats_name",
    "ats_email": "ats_email",
    "ispartner": "ispartner",
    "next_renewal_date": "next_renewal_date",
}


def find_contact_matches(contacts: pd.DataFrame, emails: List[str]) -> pd.DataFrame:
    """Look up contacts by email address.

    Match order: exact email -> domain fallback -> personal email skip -> no match.
    Returns one row per input email, in input order.
    """
    if contacts.empty:
        return pd.DataFrame()

    results = []

    output_cols = [
        c for c in contacts.columns
        if c not in ["match type", "match score", "party_number", "party_id", "contact_id", "ats_sentiment", "normalized_account", "abbreviation", "partner_of_record_name", "line_of_business"]
    ]

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
        exact = contacts[contacts["email_address"].str.lower() == email]
        if not exact.empty:
            row = exact.iloc[0]
            out = {"input": user_input, "match type": "Exact Match", "match score": 100}
            for c in output_cols:
                out[c] = row.get(c, "")
            results.append(out)
            continue

        # Domain match — skip personal/generic email providers
        domain = email.split("@")[-1]
        if domain in GENERIC_DOMAINS:
            out = {"input": user_input, "match type": "Skipped - Personal Email", "match score": 0}
            for c in output_cols:
                out[c] = ""
            results.append(out)
            continue

        domain_match = contacts[contacts["email_domain"].str.lower() == domain]

        if not domain_match.empty:
            # Pick the account with the most contacts on this domain
            top_account = domain_match.groupby("customer_id").size().idxmax()
            row = domain_match[domain_match["customer_id"] == top_account].iloc[0].copy()

            # Blank personal fields so we don't leak wrong contact data
            for col in _PERSONAL_FIELDS:
                if col in row.index:
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

    df = pd.DataFrame(results)

    # Keep only desired columns in order
    desired_order = [c for c in _CONTACT_COLUMN_ORDER if c in df.columns]
    return df.reindex(columns=desired_order)


def find_account_matches(contacts: pd.DataFrame, inputs: List[str]) -> pd.DataFrame:
    """Match company names against the account database.

    Match order: normalized exact -> abbreviation -> fuzzy (>=90 threshold).
    Returns one row per input, in input order.
    """
    if contacts.empty:
        return pd.DataFrame()

    results = []

    # Get unique accounts for matching
    unique_accounts = contacts.drop_duplicates(subset="customer_id", keep="first")
    # Pre-built list for rfuzz_process.extractOne
    account_names_list = unique_accounts["normalized_account"].tolist()

    for raw in inputs:
        user_input = raw.strip()
        norm_input = normalize_name(user_input)

        # Empty input
        if not norm_input:
            out = {"input": user_input, "match type": "No Match", "match score": 0}
            for c in _ACCOUNT_OUTPUT_COLS:
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
            for display_col, actual_col in _ACCOUNT_OUTPUT_COLS.items():
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
                for display_col, actual_col in _ACCOUNT_OUTPUT_COLS.items():
                    out[display_col] = row.get(actual_col, "")
                results.append(out)
                continue

        # Fuzzy matching — use rfuzz_process.extractOne for speed
        match = rfuzz_process.extractOne(norm_input, account_names_list, scorer=fuzz.ratio)

        if match and match[1] >= 90:
            row = unique_accounts.iloc[match[2]]
            out = {
                "input": user_input,
                "match type": "High Confidence Match",
                "match score": float(match[1]),
            }
            for display_col, actual_col in _ACCOUNT_OUTPUT_COLS.items():
                out[display_col] = row.get(actual_col, "")
        elif match and match[1] >= 70:
            row = unique_accounts.iloc[match[2]]
            out = {
                "input": user_input,
                "match type": "Possible Match",
                "match score": float(match[1]),
            }
            for display_col, actual_col in _ACCOUNT_OUTPUT_COLS.items():
                out[display_col] = row.get(actual_col, "")
        else:
            out = {
                "input": user_input,
                "match type": "No Match",
                "match score": float(match[1]) if match else 0,
            }
            for c in _ACCOUNT_OUTPUT_COLS:
                out[c] = ""

        results.append(out)

    return pd.DataFrame(results)


def find_title_to_account_matches(
    contacts: pd.DataFrame, inputs: List[Tuple[str, str]]
) -> pd.DataFrame:
    """Match job titles to contacts within specific accounts.

    For each (account_name, job_title) pair:
      1. Match the account (>=90% threshold)
      2. Score all contacts at that account by job title similarity
      3. Return all contacts sorted by title score (best first)
    """
    if contacts.empty:
        return pd.DataFrame()

    results = []

    output_cols = [
        c for c in contacts.columns
        if c not in ["match type", "match score", "party_number", "party_id", "contact_id", "ats_sentiment", "normalized_account", "abbreviation", "partner_of_record_name", "line_of_business"]
    ]

    # Get unique accounts for matching
    unique_accounts = contacts.drop_duplicates(subset="customer_id", keep="first")
    account_names_list = unique_accounts["normalized_account"].tolist()

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

        # Find matching account — same logic as find_account_matches
        matched_account_id = None

        # Normalized Exact Match
        normalized_exact = unique_accounts[unique_accounts["normalized_account"] == norm_account]
        if not normalized_exact.empty:
            matched_account_id = normalized_exact.iloc[0]["customer_id"]

        # Abbreviation Exact Match
        elif "abbreviation" in unique_accounts.columns:
            abbr_match = unique_accounts[unique_accounts["abbreviation"] == norm_account]
            if not abbr_match.empty:
                matched_account_id = abbr_match.iloc[0]["customer_id"]

        # Fuzzy matching (>=90%)
        if matched_account_id is None:
            match = rfuzz_process.extractOne(norm_account, account_names_list, scorer=fuzz.ratio)
            if match and match[1] >= 90:
                matched_account_id = unique_accounts.iloc[match[2]]["customer_id"]

        # If no account match >=90%, return no match
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

            for _, row in account_contacts.iterrows():
                contact_title = str(row.get("job_title", "")).lower()
                if contact_title:
                    title_score = fuzz.ratio(job_title_lower, contact_title)
                    scored_contacts.append((title_score, row))

            if scored_contacts:
                # Sort by score descending (best matches first)
                scored_contacts.sort(key=lambda x: x[0], reverse=True)

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
            out = {
                "input_account": account_name,
                "input_job_title": job_title,
                "match type": "No Job Title Column",
                "match score": 0,
            }
            for c in output_cols:
                out[c] = ""
            results.append(out)

    # Format output
    df = pd.DataFrame(results)
    desired_order = [
        "input_account", "input_job_title", "match type", "match score",
        "customer_name", "customer_id", "first_name", "last_name",
        "job_title", "email_address", "account_segmentation", "country",
        "prime_geo", "overlay_geo", "level15_territory_name", "arr", "ae_name",
        "ats_name", "ats_email", "ispartner",
        "next_renewal_date",
    ]
    desired_order = [c for c in desired_order if c in df.columns]
    return df.reindex(columns=desired_order)
