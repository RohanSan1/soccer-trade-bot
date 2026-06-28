#!/usr/bin/env bash
# Terminate OVH GPU instance after training completes.
#
# Usage:
#   ./ovh_teardown.sh [instance_id]
#
# If no instance_id provided, reads from /tmp/ovh_instance_info.json
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/ovh_teardown_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

# Validate environment
[[ -z "${OVH_APP_KEY:-}" ]] && error_exit "OVH_APP_KEY not set"
[[ -z "${OVH_APP_SECRET:-}" ]] && error_exit "OVH_APP_SECRET not set"
[[ -z "${OVH_CONSUMER_KEY:-}" ]] && error_exit "OVH_CONSUMER_KEY not set"
[[ -z "${OVH_PROJECT_ID:-}" ]] && error_exit "OVH_PROJECT_ID not set"

# Get instance ID from argument or file
INSTANCE_ID="${1:-}"
if [[ -z "$INSTANCE_ID" ]]; then
    INFO_FILE="/tmp/ovh_instance_info.json"
    if [[ -f "$INFO_FILE" ]]; then
        INSTANCE_ID=$(jq -r '.instance_id' "$INFO_FILE")
        log "Read instance ID from $INFO_FILE: $INSTANCE_ID"
    else
        error_exit "No instance ID provided and no info file found"
    fi
fi

log "Terminating OVH instance: $INSTANCE_ID"

# Delete instance
HTTP_CODE=$(curl -s \
    -o /dev/null \
    -w "%{http_code}" \
    -X DELETE \
    -H "X-OVH-Application: $OVH_APP_KEY" \
    -H "X-OVH-Consumer: $OVH_CONSUMER_KEY" \
    "https://api.ovh.com/1.0/cloud/project/$OVH_PROJECT_ID/instance/$INSTANCE_ID")

if [[ "$HTTP_CODE" == "204" || "$HTTP_CODE" == "200" ]]; then
    log "Instance $INSTANCE_ID terminated successfully"
else
    log "Warning: DELETE returned HTTP $HTTP_CODE (instance may not exist)"
fi

# Clean up info file
rm -f /tmp/ovh_instance_info.json

log "========================================="
log "Teardown complete"
log "========================================="
