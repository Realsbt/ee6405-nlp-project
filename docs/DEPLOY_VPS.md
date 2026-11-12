# Deploying Text Lab on a small VPS

The full demo needs torch plus a 500 MB transformer per model. That does not fit
in 1 GiB of RAM. The TF-IDF model does: it needs `joblib`, `numpy` and
`scikit-learn` only, about **130 MiB resident**, and roughly **1 ms** per
prediction.

This guide deploys the TF-IDF-only server. RoBERTa and RACE are reported as
unavailable, so the page disables them instead of failing at request time.

## What you get

| | |
|---|---|
| URL | `http://<server-ip>:8766/` |
| Models | TF-IDF + logistic regression |
| Memory | ~130 MiB resident |
| Latency | ~1 ms per prediction |
| Persistence | systemd unit, restarts on boot and on crash |

No domain name is required. A bare IP address works.

## Requirements

- Ubuntu 24.04 (or any systemd distribution with Python 3.11+)
- 1 GiB RAM, 2 vCPU is enough
- Outbound HTTPS so the script can clone the repository and download the model
- An inbound firewall rule for the port you choose

## Deploy

```bash
sudo bash scripts/deploy_tfidf_vps.sh
```

The script installs packages, creates a `textlab` service user, clones the
repository into `/opt/textlab`, builds a virtual environment with only the three
needed libraries, downloads the verified TF-IDF model from the `resources-v1`
release, and installs a hardened systemd unit.

Override defaults with environment variables:

```bash
sudo APP_DIR=/srv/textlab PORT=8080 EE6405_STORAGE=/srv/textlab-data \
  bash scripts/deploy_tfidf_vps.sh
```

## Open the port

The service binds `0.0.0.0`, but the cloud firewall usually blocks it. For Azure,
add an inbound rule for TCP 8766 in the network security group. For a local
`ufw`:

```bash
sudo ufw allow 8766/tcp
```

Then browse to `http://<server-public-ip>:8766/`.

## Verify

```bash
curl http://127.0.0.1:8766/api/health
curl -X POST http://127.0.0.1:8766/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"tfidf","text":"Human writers compose a personal story."}'
systemctl status textlab | grep Memory
```

## Operate

```bash
sudo systemctl restart textlab     # after a git pull
sudo journalctl -u textlab -f      # live logs
sudo systemctl stop textlab
```

To update the code:

```bash
sudo git -C /opt/textlab fetch --depth 1 origin main
sudo git -C /opt/textlab reset --hard origin/main
sudo systemctl restart textlab
```

## Adding a domain later

Without a domain, the site is reachable by IP over plain HTTP. If you later get
one, put a reverse proxy in front so TLS and the port number are handled for you:

```bash
sudo apt-get install -y caddy
sudo tee /etc/caddy/Caddyfile >/dev/null <<'EOF'
textlab.example.com {
    reverse_proxy 127.0.0.1:8766
}
EOF
sudo systemctl reload caddy
```

Caddy obtains a certificate automatically. Then bind the service to loopback
only by editing `--host 0.0.0.0` to `--host 127.0.0.1` in
`/etc/systemd/system/textlab.service` and running `systemctl daemon-reload`.

## Security notes

- The service has no authentication. Anyone who can reach the port can use it.
  That is fine for a course demo; add a proxy with basic auth if you need more.
- `ProtectSystem=strict`, `ProtectHome=true` and `NoNewPrivileges=true` limit what
  the process can touch. Only `EE6405_STORAGE` is writable.
- The model file is hash-checked against `experiments/reports/tfidf_lr_v1/result.json`
  at startup, so a corrupted download fails loudly instead of scoring wrongly.
- Request bodies are capped at 50 KB and text at 12,000 characters.

## Why not the full demo

| Model | Resident memory | Fits in 1 GiB |
|---|---|---|
| TF-IDF | ~130 MiB | yes |
| RoBERTa | ~2.1 GiB | no |
| RACE | ~2.1 GiB | no |
| RACE · no SupCon | ~2.7 GiB | no |

The transformer models also need the RST parser subprocess for RACE, which adds
its own footprint. A 4 GiB instance would run RoBERTa; RACE needs more headroom
and a longer first-request budget because the parser has to start.
