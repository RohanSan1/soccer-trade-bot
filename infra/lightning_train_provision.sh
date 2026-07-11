#!/usr/bin/env bash
# Provision Lightning AI Studio for training.
#
# Usage:
#   ./lightning_train_provision.sh [studio_name] [machine_type]
#
# Environment variables required:
#   LIGHTNING_USER_ID, LIGHTNING_API_KEY
#
# Machine types: "GPU" (T4), "GPU+" (A10G), "GPU++" (A100), "CPU" (default)
set -euo pipefail

STUDIO_NAME="${1:-soccer-trade-training}"
MACHINE_TYPE="${2:-GPU+}"  # GPU+ = A10G, GPU = T4, GPU++ = A100
TEAMSPACE="${LIGHTNING_TEAMSPACE:-default}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

# Validate environment
[[ -z "${LIGHTNING_USER_ID:-}" ]] && error_exit "LIGHTNING_USER_ID not set"
[[ -z "${LIGHTNING_API_KEY:-}" ]] && error_exit "LIGHTNING_API_KEY not set"

export LIGHTNING_USER_ID
export LIGHTNING_API_KEY

log "Provisioning Lightning AI Studio for TRAINING: $STUDIO_NAME"
log "Machine type: $MACHINE_TYPE"

# Check if lightning-cli is available
if ! command -v lightning &> /dev/null; then
    log "Installing lightning-cli..."
    pip install --upgrade lightning-cli 2>/dev/null || true
fi

# Create or connect to Studio with GPU
log "Starting Studio: $STUDIO_NAME on $MACHINE_TYPE"
lightning run studio \
    --name "$STUDIO_NAME" \
    --team "$TEAMSPACE" \
    --machine "$MACHINE_TYPE" \
    --disk 100 2>&1 | tee -a /tmp/lightning_train_studio.log

STUDIO_URL="https://lightning.ai/${LIGHTNING_USER_ID}/studios/${STUDIO_NAME}"

log "========================================="
log "Training Studio provisioned!"
log "Studio: $STUDIO_NAME"
log "URL:    $STUDIO_URL"
log "Machine: $MACHINE_TYPE"
log "========================================="