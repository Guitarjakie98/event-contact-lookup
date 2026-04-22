#!/usr/bin/env bash
# Bootstrap a fresh Ubuntu 24.04 Vultr server for event-contact-lookup.
# Idempotent: re-running on a working server is safe.
# Run from your laptop after a Vultr reinstall (with your SSH key attached).

set -euo pipefail

SERVER="${SERVER:-linuxuser@66.42.120.99}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REPO_URL="${REPO_URL:-https://github.com/Guitarjakie98/event-contact-lookup.git}"
APP_DIR="${APP_DIR:-/home/linuxuser/event-contact-lookup}"
DATA_DIR="${DATA_DIR:-/home/linuxuser/master_data}"
SERVICE="${SERVICE:-event-contact-lookup.service}"
PORT="${PORT:-8501}"
SWAP_GB="${SWAP_GB:-8}"

echo "==> connecting to $SERVER"
ssh -i "$SSH_KEY" "$SERVER" bash -s <<EOF
set -euo pipefail

echo "==> apt deps"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv git

echo "==> swap (${SWAP_GB}G if missing)"
if ! sudo swapon --show | grep -q swapfile; then
  sudo fallocate -l ${SWAP_GB}G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi
swapon --show

echo "==> clone or update repo"
if [ -d "$APP_DIR/.git" ]; then
  cd "$APP_DIR" && git pull
else
  git clone --depth=1 "$REPO_URL" "$APP_DIR"
fi

echo "==> venv + requirements"
cd "$APP_DIR"
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "==> data dir"
mkdir -p "$DATA_DIR"

echo "==> systemd unit"
sudo tee /etc/systemd/system/$SERVICE >/dev/null <<UNIT
[Unit]
Description=Event Contact Lookup (Streamlit)
After=network.target

[Service]
Type=simple
User=linuxuser
WorkingDirectory=$APP_DIR
Environment=DATA_DIR=$DATA_DIR
ExecStart=$APP_DIR/venv/bin/streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE

echo "==> firewall (open $PORT)"
sudo ufw allow $PORT/tcp || true

echo "==> start service"
sudo systemctl restart $SERVICE
sleep 3
systemctl is-active $SERVICE
EOF

echo
echo "==> bootstrap complete"
echo "==> next: run ./deploy/refresh-data.sh to upload the parquet"
