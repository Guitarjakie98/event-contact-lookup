# server.py

import pandas as pd
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

from matching import (
    prepare_contacts,
    find_contact_matches,
    find_account_matches,
    find_title_to_account_matches,
)

# ============================================================
# App
# ============================================================

app = FastAPI(title="Event Contact Lookup API")

# ============================================================
# Load Data Once
# ============================================================

from config import PARQUET_PATH
contacts = prepare_contacts(pd.read_parquet(PARQUET_PATH))

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
    df = find_contact_matches(contacts, req.emails)
    return df.to_dict(orient="records")

@app.post("/account_match")
def account_match(req: AccountRequest):
    df = find_account_matches(contacts, req.inputs)
    return df.to_dict(orient="records")

@app.post("/title_to_account")
def title_to_account(req: TitleAccountRequest):
    pairs = [(x[0], x[1]) for x in req.inputs]
    df = find_title_to_account_matches(contacts, pairs)
    return df.to_dict(orient="records")
