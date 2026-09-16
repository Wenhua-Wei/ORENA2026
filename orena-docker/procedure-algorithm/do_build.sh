#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="procedure-algorithm"

# Build the Docker image.
#
# Important connection to the Dockerfile:
#   - The final argument, "$SCRIPT_DIR", is the Docker build context.
#   - Because no --file/-f option is supplied, Docker automatically reads
#     "$SCRIPT_DIR/Dockerfile".
#   - COPY commands inside the Dockerfile can only access files inside this
#     build context, such as inference.py, requirements.txt, and resources/.
#
# --platform=linux/amd64 builds for the architecture used by the challenge.
# --tag gives the image the name "segment-algorithm".
# 2>&1 merges stderr into stdout so all build logs appear in one stream.

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG"  \
  "$SCRIPT_DIR" 2>&1
