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
]

# BigQuery export rename map
BQ_RENAME = {
    "account_name": "customer_name",
    "account_country": "country",
    "is_partner": "ispartner",
    "has_partner": "ispartner",
}

OUTPUT_NAME = "ContactDataApp2.1.parquet"
DATA_DIR = Path(__file__).parent / "data"
SUPPORTED_EXTS = {".csv", ".xlsx", ".xls"}


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

    output_path = Path(__file__).parent / OUTPUT_NAME
    df.to_parquet(output_path, index=False)

    print(f"\nWrote {len(df):,} rows x {len(df.columns)} columns -> {output_path}")
    print("Restart the server to load the new data.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
