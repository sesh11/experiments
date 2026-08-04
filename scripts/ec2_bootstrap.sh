#!/usr/bin/env bash
#
# One-time setup for a FRESH x86_64 Ubuntu EC2 instance so it can run the full
# stack: the orchestrator + agent runtimes (fusion / Stirrup / pi) on the box,
# with the official SWE-bench Docker harness for scoring. Run this once,
# re-login, then use ./confirm_scoring.sh or eval.run_eval as usual.
#
#   Why EC2: SWE-bench images are x86_64. Apple Silicon scores them under
#   emulation — it works but is slow and flaky; an Intel/AMD box is the fast,
#   reliable path. Nothing of OUR code runs inside Docker; the daemon is only
#   used by the scoring step.
#
# Usage (on the EC2 instance, from the repo root):
#   sudo bash scripts/ec2_bootstrap.sh
#   # then LOG OUT and back in (so the docker group applies), and:
#   cp .env.example .env && nano .env      # paste your ANTHROPIC_API_KEY
#   ./confirm_scoring.sh 3 0               # $0 gold check — expect 3/3
#
set -euo pipefail

# The person who ran sudo (so we add the right user to the docker group and
# build the venv under their ownership).
TARGET_USER="${SUDO_USER:-$USER}"
REPO_ROOT="$(pwd)"

echo "==> 1/6 Architecture check (this is the whole reason we're on EC2)"
ARCH="$(uname -m)"
if [ "$ARCH" != "x86_64" ]; then
  echo "!! uname -m = '$ARCH', not x86_64." >&2
  echo "!! You launched an ARM (Graviton) instance — it scores under emulation," >&2
  echo "!! same as the Mac. Terminate it and launch an INTEL/AMD instance" >&2
  echo "!! (e.g. m6i / c6i / t3), then re-run." >&2
  exit 1
fi
echo "    OK: x86_64. SWE-bench will score natively here."

echo "==> 2/6 Disk space check (Docker images are large)"
# Docker stores images under /var/lib/docker on the root volume by default.
AVAIL_GB="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
echo "    Free space on / : ${AVAIL_GB}G"
if [ "${AVAIL_GB:-0}" -lt 40 ]; then
  echo "!! Less than 40G free. SWE-bench images can eat tens of GB." >&2
  echo "!! Launch with a bigger root EBS volume (100-160G recommended) or add disk," >&2
  echo "!! otherwise scoring will fail partway with 'no space left on device'." >&2
  echo "!! Continuing anyway, but you've been warned." >&2
fi

echo "==> 3/6 Installing system packages (docker, python, git, node 22)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq docker.io python3 python3-venv python3-pip git curl >/dev/null
# Node 22 for the pi runtime (Ubuntu's default nodejs is too old).
if ! command -v node >/dev/null || [ "$(node -e 'console.log(process.versions.node.split(".")[0])')" -lt 20 ]; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null 2>&1
  apt-get install -y -qq nodejs >/dev/null
fi
echo "    installed: $(docker --version | cut -d, -f1), $(python3 --version), node $(node --version), $(git --version)"

echo "==> 4/6 Enabling Docker + granting '${TARGET_USER}' docker access"
systemctl enable --now docker >/dev/null 2>&1 || true
if ! getent group docker >/dev/null; then groupadd docker; fi
usermod -aG docker "$TARGET_USER"
# Prove the daemon is up (as root; the user needs a re-login for group perms).
if docker info >/dev/null 2>&1; then
  echo "    Docker daemon is running."
else
  echo "!! Docker daemon did not come up. Try: sudo systemctl status docker" >&2
fi

echo "==> 5/6 Python venv + repo requirements (as ${TARGET_USER})"
sudo -u "$TARGET_USER" bash -c "
  cd '$REPO_ROOT'
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
"
echo "    venv ready: $REPO_ROOT/.venv (includes stirrup[litellm] + swebench)"

echo "==> 6/6 pi coding agent (baseline-pi runtime)"
npm install -g --quiet @mariozechner/pi-coding-agent >/dev/null
echo "    pi $(pi --version) installed at $(command -v pi)"

cat <<EOF

============================================================================
NEXT STEPS (important — the docker group only applies after a re-login):

  1) Log out and back in   (exit your ssh session and reconnect)
        exit
        ssh ...            # reconnect to the instance

  2) Verify docker works WITHOUT sudo:
        docker run --rm hello-world

  3) Add your API key:
        cd $REPO_ROOT
        cp .env.example .env
        nano .env          # paste ANTHROPIC_API_KEY=sk-ant-...

  4) FREE checks (no LLM spend):
        source .venv/bin/activate
        python scripts/smoke_workspace.py                       # sandbox loop
        python -m eval.selftest_scoring --backend docker --limit 3   # gold 3/3

  5) When gold is 3/3, the runtime parity slice (~\$7):
        set -a; . ./.env; set +a
        python -m eval.run_eval --source swebench --backend docker \\
          --limit 3 --variants baseline-fusion baseline-stirrup baseline-pi \\
          --per-task 1.5 --budget 12 --max-steps 30 --no-judge

  6) Or the classic falsifying run (frontier_only vs scout):
        ./run_swebench.sh 15 25

NOTE ON PARALLELISM: eval.run_eval is parallel by default (4 agent workers,
2 Docker score workers) — tuned for a t3.large (2 vCPU / 8 GiB). --score-workers
is MEMORY-bound: 8 GiB tops out near 2; raise it only with more RAM. On a
burstable T3 instance, two Docker workers pinning both vCPUs for minutes will
drain CPU credits (billed as surplus in the default 'unlimited' mode) — for
sustained scoring prefer a non-burstable box (m5/c5). Pass --sequential to
eval.run_eval for the old fully-serial behavior.
============================================================================
EOF
