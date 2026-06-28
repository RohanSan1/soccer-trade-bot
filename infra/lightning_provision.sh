#!/usr/bin/env bash
# Provision Lightning AI Studio for inference.
#
# Usage:
#   ./lightning_provision.sh [studio_name]
#
# Environment variables required:
#   LIGHTNING_USER_ID, LIGHTNING_API_KEY
#
set -euo pipefail

STUDIO_NAME="${1:-soccer-trade-inference}"
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

log "Provisioning Lightning AI Studio: $STUDIO_NAME"

# Check if lightning-cli is available
if ! command -v lightning &> /dev/null; then
    log "Installing lightning-cli..."
    pip install --upgrade lightning-cli 2>/dev/null || true
fi

# Create or connect to Studio
log "Starting Studio: $STUDIO_NAME"
lightning run studio \
    --name "$STUDIO_NAME" \
    --team "$TEAMSPACE" \
    --machine "T4" \
    --disk 50 2>&1 | tee -a /tmp/lightning_studio.log

STUDIO_URL="https://lightning.ai/${LIGHTNING_USER_ID}/studios/${STUDIO_NAME}"

log "========================================="
log "Studio provisioned!"
log "Studio: $STUDIO_NAME"
log "URL:    $STUDIO_URL"
log "========================================="
