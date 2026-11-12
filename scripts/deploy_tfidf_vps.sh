#!/usr/bin/env bash
# Deploy the TF-IDF-only Text Lab on a small Ubuntu VPS.
#
# Tested target: Ubuntu 24.04, 2 vCPU, 1 GiB RAM. The full demo needs torch and
# a 500 MB transformer per model, which does not fit; TF-IDF needs about 130 MiB.
#
# Usage:
#   sudo bash scripts/deploy_tfidf_vps.sh
#
# Re-run after a git pull to restart the service.
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/textlab}"
APP_USER="${APP_USER:-textlab}"
PORT="${PORT:-8766}"
REPO_URL="${REPO_URL:-https://github.com/Realsbt/ee6405-nlp-project.git}"
STORAGE="${EE6405_STORAGE:-/var/lib/textlab/resources}"

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo bash scripts/deploy_tfidf_vps.sh" >&2
  exit 1
fi

echo "==> Installing system packages"
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git curl

echo "==> Creating service user ${APP_USER}"
id -u "$APP_USER" >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"

echo "==> Fetching the repository into ${APP_DIR}"
if [[ -d "$APP_DIR/.git" ]]; then
  git -C "$APP_DIR" fetch --depth 1 origin main
  git -C "$APP_DIR" reset --hard origin/main
else
  mkdir -p "$APP_DIR"
  git clone --depth 1 "$REPO_URL" "$APP_DIR"
fi

echo "==> Creating the Python environment"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
# Only what the TF-IDF server imports: no torch, no transformers.
"$APP_DIR/.venv/bin/pip" install --quiet joblib numpy scikit-learn

echo "==> Preparing the resource directory ${STORAGE}"
mkdir -p "$STORAGE/experiments/models"
MODEL="$STORAGE/experiments/models/tfidf_lr_v1.joblib"
if [[ ! -f "$MODEL" ]]; then
  echo "    Downloading the TF-IDF model from the resources-v1 release"
  curl -fL --retry 3 -o "$MODEL" \
    "https://github.com/Realsbt/ee6405-nlp-project/releases/download/resources-v1/tfidf_lr_v1.joblib"
fi
chown -R "$APP_USER":"$APP_USER" "$STORAGE" "$APP_DIR"

echo "==> Writing the systemd unit"
cat >/etc/systemd/system/textlab.service <<UNIT
[Unit]
Description=Text Lab (TF-IDF only)
After=network.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=EE6405_STORAGE=${STORAGE}
Environment=PYTHONUTF8=1
Environment=PYTHONDONTWRITEBYTECODE=1
ExecStart=${APP_DIR}/.venv/bin/python -B -u -m experiments.tfidf_server --host 0.0.0.0 --port ${PORT}
Restart=always
RestartSec=3
# The model is read-only at runtime; keep the service unable to write anywhere else.
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${STORAGE}

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now textlab
systemctl restart textlab

echo "==> Waiting for the service"
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${PORT}/api/health" || {
  echo "Service did not start; inspect: journalctl -u textlab -n 50" >&2
  exit 1
}

echo
echo "==> Text Lab is running on port ${PORT}"
echo "    Local check:  curl http://127.0.0.1:${PORT}/api/health"
echo "    Logs:         journalctl -u textlab -f"
echo "    Memory:       systemctl status textlab | grep Memory"
echo
echo "Next: open the port in your cloud firewall, then browse to"
echo "      http://<server-public-ip>:${PORT}/"
