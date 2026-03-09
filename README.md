# event-contact-lookup

A two-tier web app for matching event attendee contacts against the Citrix customer database. The FastAPI backend loads the contact parquet at startup and handles all matching logic; the Streamlit UI calls it over HTTP and renders results with CSV export.

---

## How to Run

**Start the API server** (port 8000):
```bash
uvicorn server:app --reload
```

**Start the UI** (port 8501):
```bash
streamlit run app.py
```

Or use the provided convenience scripts:
- **Windows:** `run_app.bat`
- **macOS:** `run_app.command`

The UI checks API health on load and shows a banner if the server isn't reachable.

---

## Features

| Tab | Input | Logic |
|---|---|---|
| **Contact Lookup** | Emails (one per line) | Exact match on `email_address` |
| **Account Match** | Company names (one per line) | RapidFuzz `fuzz.ratio` ≥ 90% against normalized account names |
| **Title to Account** | Account + job title (tab or comma separated) | Account match first, then fuzzy title score across all contacts at that account |

All three tabs return a sortable dataframe and a CSV download button.

### Personal Email Filtering

Contacts with personal/generic email domains (Gmail, Yahoo, Hotmail, Outlook, iCloud, etc.) are excluded from all outputs automatically:

- **Tabs 2 & 3** — filtered at server load time; personal-domain contacts never appear in account or title results
- **Tab 1** — if you paste a personal email address, the result shows `Skipped - Personal Email` instead of `No Match`

~70 domains are blocked, covering major providers worldwide (Google, Microsoft, Yahoo, Apple, ProtonMail, ISPs, and regional providers).

---

## Data Files

| File | Role |
|---|---|
| `ContactDataApp2.1.parquet` | Main contact/account database — loaded by `server.py` at startup |

---

## Docker (server only)

```bash
docker build -t event-contact-lookup .
docker run -p 8080:8080 event-contact-lookup
```

The Dockerfile runs `uvicorn` on port 8080. Streamlit must be run separately — it is not included in the container.
