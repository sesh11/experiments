#!/usr/bin/env bash
#
# One-time setup for a FRESH x86_64 Ubuntu EC2 instance so it can score
# SWE-bench with the official Docker harness. Run this once, re-login, then use
# ./confirm_scoring.sh as usual.
#
#   Why EC2: SWE-bench is only reliable on x86_64 Linux. An Apple Silicon Mac
#   emulates x86 and even gold patches fail to score — this box fixes that.
#
# Usage (on the EC2 instance, from the repo root):
#   sudo bash scripts/ec2_bootstrap.sh
#   # then LOG OUT and back in (so the docker group applies), and:
#   cp .env.example .env && nano .env      # paste your ANTHROPIC_API_KEY
#   ./confirm_scoring.sh 5 0               # $0 gold check — expect 5/5
#
set -euo pipefail

# The person who ran sudo (so we add the right user to the docker group).
TARGET_USER="${SUDO_USER:-$USER}"

echo "==> 1/5 Architecture check (this is the whole reason we're on EC2)"
ARCH="$(uname -m)"
if [ "$ARCH" != "x86_64" ]; then
  echo "!! uname -m = '$ARCH', not x86_64." >&2
  echo "!! You launched an ARM (Graviton) instance — it has the SAME problem as the Mac." >&2
  echo "!! Terminate it and launch an INTEL/AMD instance (e.g. m6i / c6i / t3), then re-run." >&2
  exit 1
fi
echo "    OK: x86_64. SWE-bench will score reliably here."

echo "==> 2/5 Disk space check (Docker images are large)"
# Docker stores images under /var/lib/docker on the root volume by default.
AVAIL_GB="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
echo "    Free space on / : ${AVAIL_GB}G"
if [ "${AVAIL_GB:-0}" -lt 40 ]; then
  echo "!! Less than 40G free. SWE-bench images can eat tens of GB." >&2
  echo "!! Launch with a bigger root EBS volume (100-160G recommended) or add disk," >&2
  echo "!! otherwise scoring will fail partway with 'no space left on device'." >&2
  echo "!! Continuing anyway, but you've been warned." >&2
fi

echo "==> 3/5 Installing system packages (docker, python, git)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq docker.io python3 python3-venv python3-pip git >/dev/null
echo "    installed: $(docker --version | cut -d, -f1), $(python3 --version), $(git --version)"

echo "==> 4/5 Enabling Docker + granting '${TARGET_USER}' docker access"
systemctl enable --now docker >/dev/null 2>&1 || true
if ! getent group docker >/dev/null; then groupadd docker; fi
usermod -aG docker "$TARGET_USER"
# Prove the daemon is up (as root; the user needs a re-login for group perms).
if docker info >/dev/null 2>&1; then
  echo "    Docker daemon is running."
else
  echo "!! Docker daemon did not come up. Try: sudo systemctl status docker" >&2
fi

echo "==> 5/5 Done."
cat <<EOF

============================================================================
NEXT STEPS (important — the docker group only applies after a re-login):

  1) Log out and back in   (exit your ssh session and reconnect)
        exit
        ssh ...            # reconnect to the instance

  2) Verify docker works WITHOUT sudo:
        docker run --rm hello-world

  3) Add your API key and run the FREE gold self-test:
        cd $(pwd)
        cp .env.example .env
        nano .env          # paste ANTHROPIC_API_KEY=sk-ant-...
        ./confirm_scoring.sh 5 0     # expect 'gold resolved: 5/5'

  4) When gold is 5/5, run the real, cheap comparison (~\$6):
        ./confirm_scoring.sh 5 6
============================================================================
EOF
