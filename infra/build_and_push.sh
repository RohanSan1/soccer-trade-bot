#!/usr/bin/env bash
# Build and push training Docker image to GitHub Container Registry.
#
# Usage:
#   ./infra/build_and_push.sh
#
# Prerequisites:
#   - Docker Desktop running
#   - gh auth login (GitHub CLI authenticated)
#   - docker login ghcr.io (or GITHUB_TOKEN env var set)
#
set -euo pipefail

IMAGE_NAME="ghcr.io/rohan5commit/soccer-trade-bot"
IMAGE_TAG="training"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Check GitHub CLI is authenticated
if ! gh auth status > /dev/null 2>&1; then
    echo "ERROR: GitHub CLI not authenticated. Run: gh auth login"
    exit 1
fi

# Login to GitHub Container Registry
log "Logging in to GitHub Container Registry..."
echo "${GITHUB_TOKEN:-$(gh auth token)}" | docker login ghcr.io -u "${GITHUB_USER:-rohan5commit}" --password-stdin

# Build the image
log "Building Docker image: ${FULL_IMAGE}"
docker build -t "${FULL_IMAGE}" -t "${IMAGE_NAME}:latest" "${PROJECT_DIR}"

# Push to registry
log "Pushing to GitHub Container Registry..."
docker push "${FULL_IMAGE}"
docker push "${IMAGE_NAME}:latest"

log "========================================="
log "Image pushed successfully!"
log "Image: ${FULL_IMAGE}"
log ""
log "To use in OVH AI Training:"
log "  Image: ${FULL_IMAGE}"
log "  Command: python -m model.train"
log "========================================="
