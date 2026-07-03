#!/usr/bin/env bash
# Create OVH AI Training job with custom Docker image.
#
# Usage:
#   ./infra/ovh_train.sh [ensemble|yolo|clip]
#
# This script uses the OVH AI Training API to create a job.
# Requires: OVH_APP_KEY, OVH_APP_SECRET, OVH_CONSUMER_KEY, OVH_PROJECT_ID
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/ovh_train_$(date +%Y%m%d_%H%M%S).log"

# Training mode: ensemble, yolo, or clip
TRAIN_MODE="${1:-ensemble}"

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

OVH_REGION="${OVH_REGION:-BHS5}"
IMAGE="ghcr.io/rohan5commit/soccer-trade-bot:training"

# Determine command based on training mode
case "$TRAIN_MODE" in
    ensemble)
        CMD="python -m model.train"
        FLAVOR="${OVH_FLAVOR:-h100-1}"
        ;;
    yolo)
        CMD="python -m vision.train_yolo"
        FLAVOR="${OVH_FLAVOR:-h100-1}"
        ;;
    clip)
        CMD="python -m vision.train_clip"
        FLAVOR="${OVH_FLAVOR:-h100-1}"
        ;;
    *)
        error_exit "Unknown training mode: $TRAIN_MODE (use ensemble, yolo, or clip)"
        ;;
esac

log "Creating OVH AI Training job: $TRAIN_MODE"
log "Image: $IMAGE"
log "Command: $CMD"
log "Flavor: $FLAVOR"

# Generate signature for OVH API
TIMESTAMP=$(date +%s)
OVH_API="https://api.ovh.com/1.0"

# Create the job via OVH API
JOB_ID=$(curl -s \
    -X POST \
    -H "X-OVH-Application: $OVH_APP_KEY" \
    -H "X-OVH-Consumer: $OVH_CONSUMER_KEY" \
    -H "Content-Type: application/json" \
    -d "{
        \"image\": \"$IMAGE\",
        \"command\": \"$CMD\",
        \"flavor\": \"$FLAVOR\",
        \"region\": \"$OVH_REGION\",
        \"name\": \"soccer-train-${TRAIN_MODE}-$(date +%s)\",
        \"sshPublicKeys\": [],
        \"volumes\": []
    }" \
    "${OVH_API}/cloud/project/${OVH_PROJECT_ID}/ai/job" \
    | jq -r '.id // .jobId // empty')

if [[ -z "$JOB_ID" || "$JOB_ID" == "null" ]]; then
    log "WARNING: API job creation failed or returned unexpected response."
    log "Falling back to manual instructions..."
    echo ""
    echo "========================================="
    echo "MANUAL JOB CREATION REQUIRED"
    echo "========================================="
    echo ""
    echo "1. Go to OVH AI Training console:"
    echo "   https://console.ovh.com/ai-training"
    echo ""
    echo "2. Click 'Create a job'"
    echo ""
    echo "3. Fill in:"
    echo "   - Name: soccer-train-${TRAIN_MODE}"
    echo "   - Image: ${IMAGE}"
    echo "   - Flavor: ${FLAVOR}"
    echo "   - Region: ${OVH_REGION}"
    echo ""
    echo "4. IMPORTANT: In the 'Advanced' section, add volume:"
    echo "   - Name: training-data"
    echo "   - Container: soccer-trade-bot"
    echo "   - Cache: enabled"
    echo ""
    echo "5. Submit the job"
    echo ""
    echo "6. When job is running, access shell:"
    echo "   ovhai job shell <job-id>"
    echo ""
    echo "7. Run training:"
    echo "   ${CMD}"
    echo ""
    echo "========================================="
    exit 0
fi

log "Job created successfully!"
log "Job ID: $JOB_ID"

# Write job info
cat > /tmp/ovh_training_job.json <<EOF
{
    "job_id": "$JOB_ID",
    "train_mode": "$TRAIN_MODE",
    "image": "$IMAGE",
    "command": "$CMD",
    "flavor": "$FLAVOR",
    "region": "$OVH_REGION",
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

log "Job info saved to: /tmp/ovh_training_job.json"
log ""
log "To monitor job status:"
log "  ovhai job get $JOB_ID"
log ""
log "To access job shell when running:"
log "  ovhai job shell $JOB_ID"
