# Operations

Operational reference for the event-contact-lookup deployment. Everything you need to refresh data, restart the service, or rebuild the box from scratch.

## Server

| | |
|---|---|
| **Provider** | Vultr (Chicago, region `ord`) |
| **Instance ID** | `0418bd04-1e01-4548-897c-feda784be314` |
| **Label** | `streamlit-server` |
| **Public IP** | `66.42.120.99` |
| **Plan** | `vc2-4c-8gb` (4 vCPU, 8 GB RAM, 160 GB SSD) |
| **OS** | Ubuntu 24.04 LTS |
| **Swap** | 8 GB at `/swapfile` (configured to prevent OOM kills) |
| **SSH user** | `linuxuser` (sudo without password) |
| **App URL** | http://66.42.120.99:8501 (password gate — see `check_password()` in `app.py`) |

## SSH

```bash
ssh -i ~/.ssh/id_ed25519 linuxuser@66.42.120.99
```

The Vultr account also has `~/.ssh/droplet_key` registered (different key, attached at instance creation). Either local key works for auth.

## Service

| | |
|---|---|
| **systemd unit** | `event-contact-lookup.service` |
| **Repo path** | `/home/linuxuser/event-contact-lookup` |
| **venv path** | `/home/linuxuser/event-contact-lookup/venv` |
| **Data dir** | `/home/linuxuser/master_data` (set via `DATA_DIR` env var) |
| **Parquet** | `/home/linuxuser/master_data/ContactDataApp2.1.parquet` |
| **Port** | 8501 (open in UFW, bound to `0.0.0.0`) |

```bash
# status / restart / logs
sudo systemctl status event-contact-lookup
sudo systemctl restart event-contact-lookup
sudo journalctl -u event-contact-lookup -f --no-pager
```

## Refresh the data

When you have a new contact export and want it live on the server:

1. Drop the new CSV/Excel into the **lookup-app** repo's `incoming/` folder (the lookup-app's `update_data.py` is what produces the parquet — see `lookup-app/README.md`).
2. Locally regenerate the parquet:
   ```bash
   cd ~/Desktop/Data\ Python\ Projects/Projects/lookup-app
   python update_data.py update
   ```
3. Push the regenerated parquet to the Vultr server:
   ```bash
   ./deploy/refresh-data.sh
   ```
   (script lives in this repo's `deploy/` folder — see below)

The script uploads the parquet, swaps it in, restarts the systemd unit, and prints health-check output.

## Redeploy the app code

If you push code changes to GitHub `main`, pick them up on the server:

```bash
ssh -i ~/.ssh/id_ed25519 linuxuser@66.42.120.99
cd ~/event-contact-lookup
git pull
sudo systemctl restart event-contact-lookup
```

If `requirements.txt` changed, also:
```bash
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart event-contact-lookup
```

## Rebuild from scratch (clean reinstall)

If the server breaks badly and you want to start fresh:

1. **Snapshot first** (insurance) — Vultr dashboard → server → Snapshots → Take Snapshot.
2. **Reinstall** — Vultr dashboard → server → Settings → Reinstall → pick Ubuntu 24.04 LTS → make sure your SSH key is selected.
3. **Wait** for status to go green (~5 min).
4. **Run the setup script:**
   ```bash
   ./deploy/setup-server.sh
   ```
   This is idempotent — installs Python/git/nginx, clones the repo, creates the venv, installs requirements, configures the 8 GB swap (if missing), opens UFW port 8501, sets up the systemd unit, and starts the service.
5. **Refresh the data** (`./deploy/refresh-data.sh`).

## Firewall

UFW is enabled. Only ports 22 (SSH) and 8501 (Streamlit) are open. To allow another port:
```bash
sudo ufw allow <port>/tcp
```

## Memory / OOM

Previously the Streamlit process got OOM-killed at ~7.7 GB RSS on an 8 GB box with no swap. Mitigations now in place:
- 8 GB `/swapfile` configured.
- Watch RSS with `systemctl status event-contact-lookup` (look at the `Memory:` line).

If memory creeps up over time and you want to recycle the process daily, add a systemd timer (not currently set up).

## Vultr API

API key lives outside the repo at `~/OneDrive/Desktop/Data Python Projects/.vultr/api_key.txt`. Used for programmatic instance management. Rotate via Vultr dashboard → Account → API.

## Lookup-app (sister deployment, Hetzner)

For reference — the related lookup-app lives on a different box:

| | |
|---|---|
| **Provider** | Hetzner |
| **Public IP** | `204.168.211.40` |
| **SSH user** | `root` (default key auth) |
| **Repo path** | `/root/lookup-app` |
| **Parquet** | `/root/master_data/ContactDataApp2.1.parquet` |
| **systemd unit** | `lookup-app.service` |
| **App URL** | (internal — bound to `127.0.0.1:8501`, fronted by reverse proxy) |
