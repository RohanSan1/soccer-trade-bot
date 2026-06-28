#!/usr/bin/env bash
# Provision OVH H100 GPU instance for model training.
#
# Usage:
#   ./ovh_provision.sh
#
# Environment variables required:
#   OVH_APP_KEY, OVH_APP_SECRET, OVH_CONSUMER_KEY, OVH_PROJECT_ID, OVH_REGION
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/ovh_provision_$(date +%Y%m%d_%H%M%S).log"

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
OVH_FLAVOR="${OVH_FLAVOR:-h100-360}"
IMAGE="Ubuntu 22.04"

log "Provisioning OVH H100 instance in $OVH_REGION..."

# Get available flavors
log "Fetching available flavors..."
FLAVOR_ID=$(curl -s \
    -X GET \
    -H "X-OVH-Application: $OVH_APP_KEY" \
    -H "X-OVH-Consumer: $OVH_CONSUMER_KEY" \
    -H "X-OVH-Signature: $(echo -n "GET+cloud/project/$OVH_ID/flavor?region=$OVH_REGION" | openssl dgst -sha1 -hmac "$OVH_APP_SECRET" | awk '{print $2}')" \
    "https://api.ovh.com/1.0/cloud/project/$OVH_PROJECT_ID/flavor?region=$OVH_REGION" \
    | jq -r ".[] | select(.name == \"$OVH_FLAVOR\") | .id" \
    | head -1)

if [[ -z "$FLAVOR_ID" || "$FLAVOR_ID" == "null" ]]; then
    error_exit "Flavor $OVH_FLAVOR not found in $OVH_REGION"
fi
log "Flavor ID: $FLAVOR_ID"

# Get available images
IMAGE_ID=$(curl -s \
    -X GET \
    -H "X-OVH-Application: $OVH_APP_KEY" \
    -H "X-OVH-Consumer: $OVH_CONSUMER_KEY" \
    "https://api.ovh.com/1.0/cloud/project/$OVH_PROJECT_ID/image?region=$OVH_REGION" \
    | jq -r ".[] | select(.name | test(\"Ubuntu 22.04\")) | .id" \
    | head -1)

if [[ -z "$IMAGE_ID" || "$IMAGE_ID" == "null" ]]; then
    error_exit "Ubuntu 22.04 image not found in $OVH_REGION"
fi
log "Image ID: $IMAGE_ID"

# Create instance
TIMESTAMP=$(date +%Y%m%d%H%M%S)
INSTANCE_NAME="soccer-train-${TIMESTAMP}"

log "Creating instance: $INSTANCE_NAME"
INSTANCE_ID=$(curl -s \
    -X POST \
    -H "X-OVH-Application: $OVH_APP_KEY" \
    -H "X-OVH-Consumer: $OVH_CONSUMER_KEY" \
    -H "Content-Type: application/json" \
    -d "{
        \"flavorId\": \"$FLAVOR_ID\",
        \"imageId\": \"$IMAGE_ID\",
        \"name\": \"$INSTANCE_NAME\",
        \"region\": \"$OVH_REGION\",
        \"monthlyBilling\": false
    }" \
    "https://api.ovh.com/1.0/cloud/project/$OVH_PROJECT_ID/instance" \
    | jq -r '.id')

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "null" ]]; then
    error_exit "Failed to create instance"
fi
log "Instance ID: $INSTANCE_ID"

# Wait for instance to be ACTIVE
log "Waiting for instance to become ACTIVE..."
MAX_WAIT=600
ELAPSED=0
while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    STATUS=$(curl -s \
        -X GET \
        -H "X-OVH-Application: $OVH_APP_KEY" \
        -H "X-OVH-Consumer: $OVH_CONSUMER_KEY" \
        "https://api.ovh.com/1.0/cloud/project/$OVH_PROJECT_ID/instance/$INSTANCE_ID" \
        | jq -r '.status')

    if [[ "$STATUS" == "ACTIVE" ]]; then
        log "Instance is ACTIVE"
        break
    fi

    log "Status: $STATUS (${ELAPSED}s elapsed)"
    sleep 10
    ELAPSED=$((ELAPSED + 10))
done

if [[ "$STATUS" != "ACTIVE" ]]; then
    error_exit "Instance did not become ACTIVE within ${MAX_WAIT}s"
fi

# Get IP address
IP_ADDRESS=$(curl -s \
    -X GET \
    -H "X-OVH-Application: $OVH_APP_KEY" \
    -H "X-OVH-Consumer: $OVH_CONSUMER_KEY" \
    "https://api.ovh.com/1.0/cloud/project/$OVH_PROJECT_ID/instance/$INSTANCE_ID" \
    | jq -r '.ipAddresses[0].ip')

log "Instance IP: $IP_ADDRESS"

# Install dependencies via cloud-init
log "Installing dependencies..."
sleep 30  # Wait for SSH to be ready

# Setup cloud-init user data
USER_DATA=$(cat <<'EOF'
#!/bin/bash
apt-get update && apt-get upgrade -y
apt-get install -y python3.11 python3-pip git ffmpeg nvidia-driver-535 nvidia-utils-535
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip3 install paddlepaddle-gpu paddleocr ultralytics transformers xgboost lightgbm scikit-learn
pip3 install statsbombpy soccerdata requests beautifulsoup4 py-clob-client websockets cryptography
pip3 install lightning-sdk paramiko python-dotenv tqdm joblib
EOF
)

curl -s \
    -X PUT \
    -H "X-OVH-Application: $OVH_APP_KEY" \
    -H "X-OVH-Consumer: $OVH_CONSUMER_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"userData\": $(echo "$USER_DATA" | jq -Rs .)}" \
    "https://api.ovh.com/1.0/cloud/project/$OVH_PROJECT_ID/instance/$INSTANCE_ID"

# Write instance info for teardown script
cat > /tmp/ovh_instance_info.json <<EOF
{
    "instance_id": "$INSTANCE_ID",
    "instance_name": "$INSTANCE_NAME",
    "ip_address": "$IP_ADDRESS",
    "region": "$OVH_REGION",
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

log "========================================="
log "Instance provisioned successfully!"
log "Instance ID: $INSTANCE_ID"
log "IP Address:  $IP_ADDRESS"
log "Region:      $OVH_REGION"
log "Instance info saved to: /tmp/ovh_instance_info.json"
log "========================================="
