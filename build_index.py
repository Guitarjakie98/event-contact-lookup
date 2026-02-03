import os
import re
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------
# CONFIG
# ----------------------------

INPUT_PARQUET = "data/ContactDataApp2.1.parquet"
OUTPUT_INDEX = "data/account_index.faiss"
OUTPUT_METADATA = "data/account_metadata.parquet"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------------------
# NORMALIZATION
# ----------------------------

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""

    name = name.lower()
    name = re.sub(r"[^a-z0-9\s()]", "", name)
    name = re.sub(
        r"\b(inc|llc|ltd|limited|corp|corporation|co|plc|sa|sarl|bv|gmbh|ag|nv)\b",
        "",
        name,
    )
    name = re.sub(r"\s+", " ", name).strip()
    return name

# ----------------------------
# MAIN
# ----------------------------

def main():
    print("Loading parquet...")
    df = pd.read_parquet(INPUT_PARQUET)
    df.columns = df.columns.str.lower().str.strip()

    if "customer_name" not in df.columns:
        raise ValueError("customer_name column not found")

    print("Normalizing names...")
    df["normalized_account"] = df["customer_name"].apply(normalize_name)

    names = df["normalized_account"].tolist()

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding embeddings...")
    embeddings = model.encode(
        names,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    print("Building FAISS index...")
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.add(embeddings)

    print("Saving index...")
    faiss.write_index(index, OUTPUT_INDEX)

    print("Saving metadata...")
    keep_cols = [
        "customer_id",
        "customer_name",
        "normalized_account",
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

    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_parquet(OUTPUT_METADATA, index=False)

    print("Done.")
    print(f"Index: {OUTPUT_INDEX}")
    print(f"Metadata: {OUTPUT_METADATA}")

if __name__ == "__main__":
    main()
