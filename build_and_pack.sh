#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="gwga-cv2-vit"
IMAGE_TAG="latest"
SQSH_PATH="/lustre/shared/containers/${IMAGE_NAME}.sqsh"
DOCKER_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo "[1/4] Building Docker image: ${DOCKER_IMAGE}"
docker build -t "${DOCKER_IMAGE}" .

echo "[2/4] Creating output directory"
sudo mkdir -p /lustre/shared/containers

echo "[3/4] Converting Docker image to Apptainer/Singularity .sqsh"
if command -v apptainer >/dev/null 2>&1; then
  apptainer build "${SQSH_PATH}" "docker-daemon:${DOCKER_IMAGE}"
elif command -v singularity >/dev/null 2>&1; then
  singularity build "${SQSH_PATH}" "docker-daemon:${DOCKER_IMAGE}"
else
  echo "Error: apptainer or singularity is not installed." >&2
  exit 1
fi

echo "[4/4] Done"
echo "Saved image at: ${SQSH_PATH}"
echo "Run with:"
echo "  apptainer exec ${SQSH_PATH} python GWGA/GWGA_single_input.py"
