import sys
import pandas as pd
from pathlib import Path

# Raw Eloqua export column names
REQUIRED_COLS_RAW = [
    "oracle account customer name",
    "oracle account customer id",
    "eloqua contacts email address",
    "eloqua contacts email address domain",
]
OPTIONAL_COLS_RAW = [
    "eloqua contacts first name",
    "eloqua contacts last name",
    "eloqua contacts job title",
    "oracle account account segmentation",
    "oracle account country",
    "oracle account line of business",
    "ae person name",
    "ae level14 territory name",
    "ats team person name",
    "arr total arr",
    "arr next renewal date",
    "is partner",
    "eloqua accounts account engagement score",
]

# Pre-cleaned / snake_case export column names
REQUIRED_COLS_CLEAN = [
    "customer_name",
    "customer_id",
    "email_address",
    "email_domain",
]
OPTIONAL_COLS_CLEAN = [
    "first_name",
    "last_name",
    "job_title",
    "account_segmentation",
    "country",
    "line_of_business",
    "ae_name",
    "level15_territory_name",
    "ats_name",
    "arr",
    "next_renewal_date",
    "ispartner",
    "partner_of_record_name",
    "account_engagement_score",
    "prime_geo",
]

# BigQuery export rename map
BQ_RENAME = {
    "account_name": "customer_name",
    "account_country": "country",
    "is_partner": "ispartner",
    "has_partner": "ispartner",
    "geo": "prime_geo",
}

OUTPUT_NAME = "ContactDataApp2.1.parquet"
DATA_DIR = Path(__file__).parent / "data"
SUPPORTED_EXTS = {".csv", ".xlsx", ".xls"}

# CX Accounts file for Prime Geo lookup
CX_ACCOUNTS_PATH = Path(__file__).parent.parent.parent / "master_data" / "CX_Accounts_4_2_26.csv"


def pick_input_file() -> Path:
    candidates = sorted(
        [f for f in DATA_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTS],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print(f"Error: no CSV or Excel file found in {DATA_DIR}")
        print("Drop your Eloqua export into the data/ folder and try again.")
        sys.exit(1)
    if len(candidates) == 1:
        return candidates[0]

    print("Files in data/:")
    for i, f in enumerate(candidates, 1):
        print(f"  [{i}] {f.name}")
    while True:
        raw = input(f"Pick a file [1-{len(candidates)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            return candidates[int(raw) - 1]
        print("  Invalid choice, try again.")


def main(input_path: str = None) -> None:
    path = Path(input_path) if input_path else pick_input_file()

    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    print(f"Reading {path.name} ...")
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        print(f"Error: unsupported file type '{ext}'. Expected .csv, .xlsx, or .xls.")
        sys.exit(1)

    df.columns = [c.strip().lower() for c in df.columns]

    # Detect format: snake_case (pre-cleaned) vs BigQuery vs raw Eloqua
    if "account_name" in df.columns and "customer_name" not in df.columns:
        fmt = "BigQuery"
        print(f"Detected format: {fmt}")
        df = df.rename(columns=BQ_RENAME)
        df["email_domain"] = df["email_address"].str.split("@").str[-1].str.lower()
        required, optional = REQUIRED_COLS_CLEAN, OPTIONAL_COLS_CLEAN
    elif "customer_name" in df.columns or "email_address" in df.columns:
        fmt = "pre-cleaned"
        print(f"Detected format: {fmt}")
        required, optional = REQUIRED_COLS_CLEAN, OPTIONAL_COLS_CLEAN
    else:
        fmt = "raw Eloqua"
        print(f"Detected format: {fmt}")
        required, optional = REQUIRED_COLS_RAW, OPTIONAL_COLS_RAW

    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        print("Error: missing critical columns — cannot write parquet.")
        for c in missing_required:
            print(f"  MISSING (required): '{c}'")
        sys.exit(1)

    for c in optional:
        if c not in df.columns:
            print(f"  Warning: optional column not found: '{c}'")

    # Merge fields from CX Accounts file
    if CX_ACCOUNTS_PATH.exists():
        print(f"Enriching from {CX_ACCOUNTS_PATH.name} ...")
        cx = pd.read_csv(CX_ACCOUNTS_PATH, low_memory=False)
        cx.columns = [c.strip().lower() for c in cx.columns]

        # Build customer_id lookup (one row per customer)
        cx_lookup = (
            cx.drop_duplicates(subset="bi customer id")
            .rename(columns={"bi customer id": "customer_id"})
        )

        # Merge Prime Geo if not already in data
        if "prime_geo" not in df.columns:
            geo_lookup = cx_lookup[["customer_id", "prime geo"]].rename(columns={"prime geo": "prime_geo"})
            df = df.merge(geo_lookup, on="customer_id", how="left")
            df["prime_geo"] = df["prime_geo"].fillna("")
            matched = (df["prime_geo"] != "").sum()
            print(f"  Matched Prime Geo for {matched:,} / {len(df):,} rows")
        else:
            df["prime_geo"] = df["prime_geo"].fillna("")
            print(f"  Prime Geo already in data ({(df['prime_geo'] != '').sum():,} rows populated)")

        # Merge ARR if not already in data
        arr_col = [c for c in cx_lookup.columns if "total arr" in c]
        if arr_col and "arr" not in df.columns:
            arr_lookup = cx_lookup[["customer_id", arr_col[0]]].rename(columns={arr_col[0]: "arr"})
            df = df.merge(arr_lookup, on="customer_id", how="left")
            df["arr"] = df["arr"].fillna("")
            matched = (df["arr"] != "").sum()
            print(f"  Matched ARR for {matched:,} / {len(df):,} rows")
        elif "arr" in df.columns:
            print(f"  ARR already in data")
        else:
            print(f"  Warning: no ARR column found in CX Accounts file")

        # Merge AE Name (Prime Territory Owner) — always overwrite from CX Accounts
        if "prime territory owner" in cx_lookup.columns:
            ae_lookup = cx_lookup[["customer_id", "prime territory owner"]].rename(columns={"prime territory owner": "ae_name_cx"})
            if "ae_name" in df.columns:
                df = df.drop(columns=["ae_name"])
            df = df.merge(ae_lookup, on="customer_id", how="left")
            df = df.rename(columns={"ae_name_cx": "ae_name"})
            df["ae_name"] = df["ae_name"].fillna("")
            matched = (df["ae_name"] != "").sum()
            print(f"  Matched AE Name for {matched:,} / {len(df):,} rows")
        else:
            print(f"  Warning: no Prime Territory Owner column found in CX Accounts file")

        # Merge Overlay Geo if not already in data
        if "overlay geo" in cx_lookup.columns and "overlay_geo" not in df.columns:
            geo_overlay_lookup = cx_lookup[["customer_id", "overlay geo"]].rename(columns={"overlay geo": "overlay_geo"})
            df = df.merge(geo_overlay_lookup, on="customer_id", how="left")
            df["overlay_geo"] = df["overlay_geo"].fillna("")
            matched = (df["overlay_geo"] != "").sum()
            print(f"  Matched Overlay Geo for {matched:,} / {len(df):,} rows")
        elif "overlay_geo" in df.columns:
            print(f"  Overlay Geo already in data")
        else:
            print(f"  Warning: no Overlay Geo column found in CX Accounts file")

        # Merge Parent/Child flag (joins on party_number, not customer_id)
        if "parent/child" in cx.columns and "parent_child" not in df.columns and "party_number" in df.columns:
            pc_lookup = (
                cx[["party number", "parent/child"]]
                .drop_duplicates(subset="party number")
                .rename(columns={"party number": "party_number", "parent/child": "parent_child"})
            )
            pc_lookup["party_number"] = pc_lookup["party_number"].astype(str)
            df["party_number"] = df["party_number"].astype(str)
            df = df.merge(pc_lookup, on="party_number", how="left")
            df["parent_child"] = df["parent_child"].fillna("")
            matched = (df["parent_child"] != "").sum()
            print(f"  Matched Parent/Child for {matched:,} / {len(df):,} rows")
        elif "parent_child" in df.columns:
            print(f"  Parent/Child already in data")
        else:
            print(f"  Warning: no Parent/Child column found in CX Accounts file")

        # Merge Next Renewal Date (Target Beachhead Quarter) if not already in data
        if "target beachhead quarter" in cx_lookup.columns and "next_renewal_date" not in df.columns:
            renewal_lookup = cx_lookup[["customer_id", "target beachhead quarter"]].rename(columns={"target beachhead quarter": "next_renewal_date"})
            df = df.merge(renewal_lookup, on="customer_id", how="left")
            df["next_renewal_date"] = df["next_renewal_date"].fillna("")
            matched = (df["next_renewal_date"] != "").sum()
            print(f"  Matched Next Renewal Date for {matched:,} / {len(df):,} rows")
        elif "next_renewal_date" in df.columns:
            print(f"  Next Renewal Date already in data")
        else:
            print(f"  Warning: no Target Beachhead Quarter column found in CX Accounts file")
    else:
        print(f"  Warning: CX Accounts file not found at {CX_ACCOUNTS_PATH}, skipping enrichment")

    output_path = Path(__file__).parent.parent.parent / "master_data" / OUTPUT_NAME
    df.to_parquet(output_path, index=False)

    print(f"\nWrote {len(df):,} rows x {len(df.columns)} columns -> {output_path}")
    print("Restart the server to load the new data.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
