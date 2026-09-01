#!/usr/bin/env bash
# Shared config/helpers for the rtx5090-connect skill scripts.
# Not meant to be run directly -- sourced by the other scripts.

RTX5090_HOST="${RTX5090_HOST:-ss@192.168.0.204}"
RTX5090_COMPOSE_DIR="${RTX5090_COMPOSE_DIR:-~/rtx5090setup/docker}"
# Host-side path (for ssh/rsync commands that run outside the container).
RTX5090_PROJECT_DIR="${RTX5090_PROJECT_DIR:-~/fusion-project}"
# Same directory as seen FROM INSIDE the train container (bind-mounted there
# at a different path than its host-side location) -- use this one for any
# command run via `docker compose exec`, never the host-side path above.
RTX5090_CONTAINER_PROJECT_DIR="${RTX5090_CONTAINER_PROJECT_DIR:-/fusion-project}"
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=6"

rtx_ssh() {
    ssh $SSH_OPTS "$RTX5090_HOST" "$@"
}

# Confirms the box at $RTX5090_HOST is actually the RTX 5090 workstation and
# not, e.g., the DGX Spark's own onboard GB10 GPU (the #1 way this fails
# silently -- a local nvidia-smi on the Spark itself always "succeeds" and
# always shows the wrong GPU).
verify_target_gpu() {
    local gpu_name
    gpu_name=$(rtx_ssh "nvidia-smi --query-gpu=name --format=csv,noheader" 2>/dev/null)
    if [ -z "$gpu_name" ]; then
        echo "FAIL: could not reach $RTX5090_HOST or nvidia-smi failed there." >&2
        print_troubleshooting
        return 1
    fi
    if [[ "$gpu_name" != *"RTX 5090"* ]]; then
        echo "FAIL: reached $RTX5090_HOST but it reports GPU '$gpu_name', not an RTX 5090." >&2
        echo "This means RTX5090_HOST is pointed at the wrong machine -- do not proceed." >&2
        return 1
    fi
    echo "OK: $RTX5090_HOST confirmed as RTX 5090 ($gpu_name)"
    return 0
}

print_troubleshooting() {
    cat >&2 <<'EOF'

Troubleshooting:
  1. Confirm the host/IP is still correct -- it's DHCP-assigned and can change.
     Ask the user for the current IP/hostname, or check the router's DHCP
     leases, then re-run with: RTX5090_HOST=ss@<new-ip> <script>
  2. Confirm you're on the same LAN (or Tailscale, if since configured) as
     the target -- this is LAN-only by default.
  3. Confirm your public key is in the target's ~/.ssh/authorized_keys.
     Password auth is disabled there, so an unauthorized key gets an
     immediate "Permission denied (publickey)" with no password fallback.
  4. For anything not covered here, read the full runbook on the target:
       ssh <host> "cat ~/rtx5090setup/dgx-spark-to-rtx5090-job-instructions.md"
EOF
}
