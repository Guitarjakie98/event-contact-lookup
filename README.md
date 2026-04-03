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
| **Account Match** | Company names (one per line) | Normalized exact → abbreviation → dual fuzzy match (`fuzz.ratio` + `fuzz.token_sort_ratio`, best of both, ≥ 90% high confidence / ≥ 80% possible) |
| **Title to Account** | Account + job title (tab or comma separated) | Account match first, then fuzzy title score across all contacts at that account |

All three tabs return a sortable dataframe and a CSV download button.

### Personal Email Filtering

Contacts with personal/generic email domains (Gmail, Yahoo, Hotmail, Outlook, iCloud, etc.) are excluded from all outputs automatically:

- **Tabs 2 & 3** — filtered at server load time; personal-domain contacts never appear in account or title results
- **Tab 1** — if you paste a personal email address, the result shows `Skipped - Personal Email` instead of `No Match`

~70 domains are blocked, covering major providers worldwide (Google, Microsoft, Yahoo, Apple, ProtonMail, ISPs, and regional providers).

---

## Updating the Data

When you have a new Eloqua/Oracle export:

1. Drop the CSV or Excel file into the `data/` folder
2. Run:
   ```bash
   python make_parquet.py
   ```
   If multiple files are in `data/`, you'll get a numbered prompt to pick one:
   ```
   Files in data/:
     [1] export_2026-03-09.xlsx
     [2] export_2026-01-15.csv
   Pick a file [1-2]:
   ```
3. The script validates required columns, then writes `ContactDataApp2.1.parquet` next to itself
4. Restart the server to load the new data:
   ```bash
   uvicorn server:app --reload
   ```

You can also pass a path directly to skip the prompt:
```bash
python make_parquet.py "path/to/export.xlsx"
```

### Input File Fields

The script accepts either **raw Eloqua exports** (verbose column names) or **pre-cleaned files** (snake_case columns). It auto-detects the format by checking whether columns like `customer_name` or `email_address` are already present. Column names are lowercased and stripped before validation.

| Raw Eloqua Column | Pre-cleaned Column | Required? | Description |
|---|---|---|---|
| `oracle account customer name` | `customer_name` | Required | Account/company display name |
| `oracle account customer id` | `customer_id` | Required | Master account identifier |
| `eloqua contacts email address` | `email_address` | Required | Contact email (exact match key) |
| `eloqua contacts email address domain` | `email_domain` | Required | Email domain (used for domain fallback) |
| `eloqua contacts first name` | `first_name` | Optional | Contact first name |
| `eloqua contacts last name` | `last_name` | Optional | Contact last name |
| `eloqua contacts job title` | `job_title` | Optional | Job title (used in Title to Account tab) |
| `oracle account account segmentation` | `account_segmentation` | Optional | Account tier (Enterprise, RM, etc.) |
| `oracle account country` | `country` | Optional | Account country |
| `oracle account line of business` | `line_of_business` | Optional | Line of business |
| `ae person name` | `ae_name` | Optional | Account executive name |
| `ae level14 territory name` | `level15_territory_name` | Optional | Territory name |
| `ats team person name` | `ats_name` | Optional | ATS person name |
| `arr total arr` | `arr` | Optional | Annual recurring revenue |
| `arr next renewal date` | `next_renewal_date` | Optional | Next renewal date |
| `is partner` / `has_partner` | `ispartner` | Optional | Partner flag |
| — | `partner_of_record_name` | Optional | Partner company name |
| `eloqua accounts account engagement score` | `account_engagement_score` | Optional | Engagement score |

---

## Runtime Data Enrichment

When the Streamlit app or server loads the parquet file, `prepare_contacts()` in `matching.py` automatically enriches the data using the CX Accounts CSV located at `master_data/CX_Accounts_4_2_26.csv` (relative to the repo's parent directory). This happens at load time — the parquet file itself is not modified.

### What gets enriched

**1. ATS Name → Overlay Territory Owner**

The CX Accounts spreadsheet's `Overlay Territory Owner` column is treated as the source of truth for the ATS rep assigned to each account. On load, the app:

- Joins the CX data to the contact data on `customer_id` (mapped from `BI Customer ID` in the CX sheet)
- Overwrites the `ats_name` field with the `Overlay Territory Owner` value for every account where the CX sheet has a value
- Accounts not present in the CX sheet keep their original `ats_name` from the parquet

This was done because the parquet's `ats_name` field uses informal display names (e.g., "Liz Kirby", "Terry Hooper") while the CX sheet has the authoritative legal/full names (e.g., "Elizabeth Kaitlyn Kirby", "Terence James Hooper"). Analysis showed 98.2% of accounts had the same person under both naming conventions, with only ~16 accounts having a genuinely different person assigned.

**2. ARR Backfill**

The CX Accounts spreadsheet's `Total ARR` column fills in missing ARR values:

- Joins on `customer_id` the same way as above
- Only fills rows where `arr` is blank or missing — existing ARR values are not overwritten

### Columns removed

The following columns were removed from the output as they are no longer needed:

- `ispartner` — partner flag
- `next_renewal_date` — renewal date

---

## Fuzzy Matching Details

Account Match and Title to Account tabs use a multi-strategy fuzzy matching pipeline:

1. **Normalized exact match** — lowercase, strip whitespace and common suffixes (Inc, Corp, Ltd, GmbH, etc.), then compare
2. **Abbreviation match** — extract uppercase initials from the input and compare against known abbreviations
3. **Dual fuzzy scoring** — run both `fuzz.ratio` (character-level) and `fuzz.token_sort_ratio` (word-order-independent), take the higher score. This handles inputs where words are reordered (e.g., "Healthcare GE" still matches "GE Healthcare" at 100%)

Results are tiered:
- **High Confidence Match** — score ≥ 90%
- **Possible Match** — score 80–89% (floor raised from 70 to reduce false positives)
- **No Match** — below 80%

Both fuzzy calls use `score_cutoff` so RapidFuzz can short-circuit hopeless candidates early.

### Accuracy (tested against 209 edge cases)

| Input variation | Precision |
|---|---|
| Exact, casing, whitespace | 100% |
| Typos (single and double char swaps) | 100% |
| Word-order swaps | 100% |
| Suffix add/remove | 100% |
| `&` vs `and` substitution | 100% |
| Dropped/added words | 75–91% |
| Truncated or single-word inputs | 40–80% |
| **Overall** | **93.2%** |

For realistic inputs (full company names with typos, casing issues, reordered words), precision is effectively 100%. The only false positives come from degenerate inputs like single words or bare acronyms.

---

## Data Files

| File | Role |
|---|---|
| `ContactDataApp2.1.parquet` | Main contact/account database — loaded by `server.py` at startup |
| `data/` | Drop zone for raw Eloqua exports (CSV or Excel) — not committed to git |
| `../../master_data/CX_Accounts_4_2_26.csv` | CX Accounts export — used at load time to enrich ATS names and backfill ARR |

---

## Docker (server only)

```bash
docker build -t event-contact-lookup .
docker run -p 8080:8080 event-contact-lookup
```

The Dockerfile runs `uvicorn` on port 8080. Streamlit must be run separately — it is not included in the container.
