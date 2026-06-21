#!/usr/bin/env bash
# PEAR — SOTA / robust bootstrap for Scaleway H100 (Ubuntu 22.04/24.04).
#
# Idempotent. Safe to re-run after a partial install. Uses `uv` for a
# fast, conflict-aware resolver; falls back to pip if uv unavailable.
#
# Pinned, no-conflict 2026-06 stack:
#   - PyTorch 2.7.0 + cu126 (Hopper SM 9.0 first-class support)
#   - flash-attention 2.7.4.post1 prebuilt wheel (torch2.7 / cu126 / cp312)
#   - transformers 4.62.1 (first stable Qwen3.5 release)
#   - All other deps pinned in pear/requirements.txt + constraints.txt
#
# Usage:
#   bash pear/setup_h100.sh                # full install
#   bash pear/setup_h100.sh --skip-torch   # skip torch (e.g. already installed)
#   bash pear/setup_h100.sh --verify       # only re-run the doctor
#
# After it finishes:
#   source .venv/bin/activate
#   python -m pear doctor          # diagnose env
#   python -m pear print-modules   # confirm vision module on Qwen3.5
#   python -m pear smoke           # ~2 min synthetic end-to-end
#   python -m pear probe           # main run
set -Eeuo pipefail

# -------------------------------------------------------------- args
SKIP_TORCH=0
VERIFY_ONLY=0
for a in "$@"; do
    case "$a" in
        --skip-torch) SKIP_TORCH=1 ;;
        --verify)     VERIFY_ONLY=1 ;;
        -h|--help)    grep '^#' "$0" | head -40 ; exit 0 ;;
        *) echo "[setup] unknown arg: $a" >&2 ; exit 2 ;;
    esac
done

# -------------------------------------------------------------- paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
PY_BIN="${VENV}/bin/python"

# -------------------------------------------------------------- versions (pinned)
TORCH_VER="2.7.0"
TVISION_VER="0.22.0"
CUDA_TAG="cu126"           # torch wheel tag (full)
FA_CUDA_TAG="cu12"        # flash-attn wheel tag (major only)
FA_VER="2.8.3"

# -------------------------------------------------------------- HF caches
# Make HF caches predictable and survivable across sessions.
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"  # fast downloads (Xet protocol)
mkdir -p "${HF_HOME}"

# -------------------------------------------------------------- log helpers
log()  { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[setup]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[setup]\033[0m FATAL: %s\n" "$*" >&2 ; exit 1; }

trap 'die "command failed at line $LINENO: ${BASH_COMMAND}"' ERR

# -------------------------------------------------------------- verify-only short path
if (( VERIFY_ONLY )); then
    [[ -x "${PY_BIN}" ]] || die "venv not found at ${VENV}; run setup_h100.sh without --verify first."
    "${PY_BIN}" -m pear doctor
    exit 0
fi

# ============================================================== PREFLIGHT
log "preflight: GPU"
command -v nvidia-smi >/dev/null || die "nvidia-smi missing. No CUDA driver."
DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
log "  device: ${GPU}   driver: ${DRV}   memory: ${MEM}"
DRV_MAJOR="${DRV%%.*}"
if (( DRV_MAJOR < 560 )); then
    warn "driver < 560 — cu126 wheels need driver >= 560. You have ${DRV}."
    warn "Consider running on an updated image, or accept that torch.cuda may fail."
fi

log "preflight: python"
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
log "  system python: ${PY_VER}"
if [[ "${PY_VER}" != "3.12" && "${PY_VER}" != "3.11" ]]; then
    warn "tested with python 3.11/3.12; got ${PY_VER}. Flash-attn wheel may not match."
fi
PY_TAG="cp${PY_VER/./}"   # cp312

log "preflight: disk"
DISK_FREE="$(df -BG "${REPO_ROOT}" | awk 'NR==2 {gsub("G",""); print $4}')"
log "  free: ${DISK_FREE}G  (need ≥ 30G: venv ~10G + Qwen3.5-2B ~6G + dataset cache ~10G)"
(( DISK_FREE >= 30 )) || warn "low disk space; downloads may fail mid-run."

# ============================================================== APT
log "apt: install system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    python3-venv python3-pip python3-dev \
    build-essential git curl ca-certificates \
    libjpeg-dev zlib1g-dev  # pillow build fallback

# ============================================================== UV
log "uv: install (fast Python resolver)"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
UV="$(command -v uv)"
log "  uv: $(${UV} --version)"

# ============================================================== VENV
if [[ ! -x "${PY_BIN}" ]]; then
    log "venv: creating at ${VENV}"
    "${UV}" venv --python "python${PY_VER}" "${VENV}"
fi
# uv picks up the venv from VIRTUAL_ENV; this is the syntax it actually supports in 0.11+.
export VIRTUAL_ENV="${VENV}"
UV_PIP=( "${UV}" pip )

log "venv: upgrade pip/wheel/setuptools"
"${UV_PIP[@]}" install --quiet --upgrade pip wheel setuptools

# ============================================================== TORCH
if (( SKIP_TORCH == 0 )); then
    log "torch: install ${TORCH_VER}+${CUDA_TAG} (Hopper-optimal)"
    "${UV_PIP[@]}" install --quiet \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
        "torch==${TORCH_VER}" "torchvision==${TVISION_VER}"
    log "torch: verifying CUDA visibility"
    "${PY_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch sees no CUDA — driver/wheel mismatch?"
cc = torch.cuda.get_device_capability(0)
print(f"  torch {torch.__version__}   cuda {torch.version.cuda}")
print(f"  device {torch.cuda.get_device_name(0)} (SM {cc[0]}.{cc[1]})")
# Quick sanity tensor op on GPU
x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16)
y = (x @ x).sum().item()
print(f"  bfloat16 matmul OK  (sum={y:.2f})")
PY
else
    log "torch: --skip-torch given, leaving as-is"
fi

# ============================================================== HF transfer (optional speedup)
log "hf_transfer: installing for faster model downloads"
"${UV_PIP[@]}" install --quiet hf_transfer || warn "hf_transfer install failed (downloads will be slower)"

# ============================================================== MAIN DEPS
log "deps: install (constrained)"
"${UV_PIP[@]}" install --quiet \
    --constraint "${REPO_ROOT}/pear/constraints.txt" \
    -r "${REPO_ROOT}/pear/requirements.txt"

# ============================================================== FLASH-ATTENTION
log "flash-attention 2: install prebuilt wheel (Hopper-optimal)"
if "${PY_BIN}" -c "import flash_attn" 2>/dev/null; then
    log "  already installed: $(${PY_BIN} -c 'import flash_attn; print(flash_attn.__version__)')"
else
    # Wheel filename format from Dao-AILab/flash-attention releases.
    ABI_TAG="cxx11abiFALSE"
    FA_WHL="flash_attn-${FA_VER}+${FA_CUDA_TAG}torch${TORCH_VER%.*}${ABI_TAG}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
    FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/${FA_WHL}"
    log "  trying: ${FA_URL}"
    if "${UV_PIP[@]}" install --quiet "${FA_URL}"; then
        log "  ok"
    else
        warn "prebuilt wheel not available — falling back to source build (10–30 min)"
        warn "if you want to skip FA entirely, re-run with --skip-torch and ignore this."
        if ! "${UV_PIP[@]}" install --quiet flash-attn --no-build-isolation; then
            warn "flash-attn install failed; PEAR will fall back to SDPA (still correct, just slower)"
        fi
    fi
fi

# ============================================================== HF cache env file
ENV_FILE="${REPO_ROOT}/.venv/bin/activate.d_pear.sh"
mkdir -p "$(dirname "${ENV_FILE}")"
cat > "${ENV_FILE}" <<EOF
# auto-sourced PEAR env vars
export HF_HOME="${HF_HOME}"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
EOF
# Tack it onto the venv activate script if not already there.
if ! grep -q "_pear.sh" "${VENV}/bin/activate" 2>/dev/null; then
    echo ". \"${ENV_FILE}\"" >> "${VENV}/bin/activate"
fi
log "env: wrote ${ENV_FILE} (auto-sourced by venv activate)"

# ============================================================== DOCTOR
log "doctor: full env diagnostic"
"${PY_BIN}" -m pear doctor

log ""
log "================================================================"
log "  SETUP COMPLETE"
log "================================================================"
log ""
log "  Activate:   source ${VENV}/bin/activate"
log "  Diagnose:   python -m pear doctor"
log "  Detect VM:  python -m pear print-modules"
log "  Smoke:      python -m pear smoke      # ~2 min synthetic"
log "  Probe:      python -m pear probe      # ~1-2 GPU-days @ defaults"
log "  Analyze:    python -m pear analyze"
log ""
