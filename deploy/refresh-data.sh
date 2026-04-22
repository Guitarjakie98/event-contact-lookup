#!/usr/bin/env bash
# Push the local parquet to the Vultr server, swap it in, restart the service.
# Run this from your laptop (Git Bash / WSL / macOS / Linux).

set -euo pipefail

# --- config (override with env vars) ---
SERVER="${SERVER:-linuxuser@66.42.120.99}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
PARQUET_LOCAL="${PARQUET_LOCAL:-$HOME/OneDrive/Desktop/Data Python Projects/master_data/ContactDataApp2.1.parquet}"
PARQUET_REMOTE="${PARQUET_REMOTE:-/home/linuxuser/master_data/ContactDataApp2.1.parquet}"
SERVICE="${SERVICE:-event-contact-lookup.service}"

if [ ! -f "$PARQUET_LOCAL" ]; then
  echo "Local parquet not found: $PARQUET_LOCAL" >&2
  exit 1
fi

echo "==> uploading $PARQUET_LOCAL"
echo "    to $SERVER:$PARQUET_REMOTE"
scp -i "$SSH_KEY" "$PARQUET_LOCAL" "$SERVER:$PARQUET_REMOTE.new"

echo "==> swapping in + restarting service"
ssh -i "$SSH_KEY" "$SERVER" "
  set -e
  cd \"$(dirname "$PARQUET_REMOTE")\"
  mkdir -p backups
  cp -p \"$(basename "$PARQUET_REMOTE")\" \"backups/$(basename "$PARQUET_REMOTE" .parquet)_backup_\$(date +%Y%m%d_%H%M%S).parquet\"
  mv \"$(basename "$PARQUET_REMOTE").new\" \"$(basename "$PARQUET_REMOTE")\"
  sudo systemctl restart $SERVICE
  sleep 3
  systemctl is-active $SERVICE
  ls -la \"$(basename "$PARQUET_REMOTE")\"
"

echo "==> done"
