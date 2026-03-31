# server.py

import pandas as pd
import re
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from rapidfuzz import fuzz, process as rfuzz_process

# ============================================================
# App
# ============================================================

app = FastAPI(title="Event Contact Lookup API")

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
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s()]", "", name)
    name = re.sub(
        r"\b(inc|llc|ltd|limited|corp|corporation|co|plc|sa|sao|sarl|bv|gmbh|ag|nv)\b",
        "",
        name,
    )
    return re.sub(r"\s+", " ", name).strip()


# ============================================================
# Load Data Once
# ============================================================

DATA_PATH = "ContactDataApp2.1.parquet"
contacts = pd.read_parquet(DATA_PATH)
contacts.columns = contacts.columns.str.lower().str.strip()

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

contacts = contacts.rename(columns=rename_map)

contacts["normalized_account"] = contacts["customer_name"].apply(normalize_name)

contacts = contacts[~contacts["email_domain"].str.lower().isin(GENERIC_DOMAINS)].reset_index(drop=True)

accounts = contacts.drop_duplicates(subset="customer_id").reset_index(drop=True)

# Pre-built list for vectorized account matching
account_names_list: list = accounts["normalized_account"].tolist()

# Hash index for O(1) email lookup
email_index = contacts.set_index("email_address")

# In-process cache: normalized account name -> result dict
_account_cache: dict = {}

# ============================================================
# Matching Logic
# ============================================================

def find_account_matches(inputs: List[str]):
    results = []

    for raw in inputs:
        norm = normalize_name(raw)
        if norm in _account_cache:
            entry = {**_account_cache[norm], "input": raw}
            results.append(entry)
            continue

        match = rfuzz_process.extractOne(norm, account_names_list, scorer=fuzz.ratio)
        if match and match[1] >= 90:
            row = accounts.iloc[match[2]]
            entry = {
                "input": raw,
                "match_type": "High Confidence Match",
                "match_score": match[1],
                "customer_name": row["customer_name"],
                "customer_id": row["customer_id"],
                "arr": row.get("arr"),
                "ae_name": row.get("ae_name"),
                "ispartner": row.get("ispartner"),
                "level15_territory_name": row.get("level15_territory_name"),
                "partner_of_record_name": row.get("partner_of_record_name"),
            }
        else:
            entry = {
                "input": raw,
                "match_type": "No Match",
                "match_score": match[1] if match else 0,
            }
        _account_cache[norm] = entry
        results.append(entry)

    return pd.DataFrame(results)


def find_contact_matches(emails: List[str]):
    results = []

    for email in emails:
        domain = email.split("@")[-1].lower() if "@" in email else ""
        if domain in GENERIC_DOMAINS:
            results.append({"input": email, "match_type": "Skipped - Personal Email"})
            continue

        if email in email_index.index:
            row = email_index.loc[email]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            results.append({
                "input": email,
                "match_type": "Exact Match",
                "customer_name": row["customer_name"],
                "email_address": row["email_address"],
                "job_title": row["job_title"],
            })
        else:
            results.append({"input": email, "match_type": "No Match"})

    return pd.DataFrame(results)


def find_title_to_account_matches(pairs):
    results = []

    for account, title in pairs:
        acc_df = find_account_matches([account])

        if acc_df.iloc[0]["match_type"] != "High Confidence Match":
            results.append({
                "input_account": account,
                "input_title": title,
                "match_type": "No Account Match",
            })
            continue

        cust_id = acc_df.iloc[0]["customer_id"]
        subset = contacts[contacts["customer_id"] == cust_id].reset_index(drop=True)
        titles_list = subset["job_title"].fillna("").str.lower().tolist()

        scored = rfuzz_process.extract(
            title.lower(), titles_list, scorer=fuzz.ratio, limit=len(titles_list)
        )
        for _match_text, score, idx in scored:
            row = subset.iloc[idx]
            results.append({
                "input_account": account,
                "input_title": title,
                "contact_name": f"{row['first_name']} {row['last_name']}",
                "job_title": row["job_title"],
                "title_score": score,
                "partner_of_record_name": row.get("partner_of_record_name"),
            })

    return pd.DataFrame(results)

# ============================================================
# Schemas
# ============================================================

class ContactRequest(BaseModel):
    emails: List[str]

class AccountRequest(BaseModel):
    inputs: List[str]

class TitleAccountRequest(BaseModel):
    inputs: List[List[str]]

# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():
    return {"service": "event-contact-lookup-api"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/contact_lookup")
def contact_lookup(req: ContactRequest):
    df = find_contact_matches(req.emails)
    return df.to_dict(orient="records")

@app.post("/account_match")
def account_match(req: AccountRequest):
    df = find_account_matches(req.inputs)
    return df.to_dict(orient="records")

@app.post("/title_to_account")
def title_to_account(req: TitleAccountRequest):
    pairs = [(x[0], x[1]) for x in req.inputs]
    df = find_title_to_account_matches(pairs)
    return df.to_dict(orient="records")
