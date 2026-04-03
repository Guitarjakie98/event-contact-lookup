import os
from pathlib import Path

# DATA_DIR env var overrides the default relative path.
# On the Vultr server this is set to /home/master_data via systemd.
# Locally it falls back to ../../master_data (relative to this file).
DATA_DIR = Path(os.environ.get(
    "DATA_DIR",
    Path(__file__).parent.parent.parent / "master_data",
))

PARQUET_PATH = str(DATA_DIR / "ContactDataApp2.1.parquet")
CX_ACCOUNTS_PATH = str(DATA_DIR / "CX_Accounts_4_2_26.csv")
