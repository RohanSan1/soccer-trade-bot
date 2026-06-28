#!/usr/bin/env bash
# Stop Lightning AI Studio after match ends.
#
# Usage:
#   ./lightning_teardown.sh [studio_name]
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

[[ -z "${LIGHTNING_USER_ID:-}" ]] && error_exit "LIGHTNING_USER_ID not set"
[[ -z "${LIGHTNING_API_KEY:-}" ]] && error_exit "LIGHTNING_API_KEY not set"

export LIGHTNING_USER_ID
export LIGHTNING_API_KEY

log "Stopping Lightning AI Studio: $STUDIO_NAME"

lightning stop studio \
    --name "$STUDIO_NAME" \
    --team "$TEAMSPACE" 2>&1 || log "Warning: Studio may not be running"

log "========================================="
log "Studio $STUDIO_NAME stopped"
log "========================================="
