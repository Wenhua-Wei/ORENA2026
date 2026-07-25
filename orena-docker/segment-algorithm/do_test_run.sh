#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="segment-algorithm"

DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

# Use the GPU when Docker's NVIDIA runtime is available; fall back to CPU
# otherwise (the dummy runs fine on CPU — the platform always provides a GPU).
GPU_FLAGS=()
if docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null | grep -qw nvidia; then
    GPU_FLAGS=(--gpus all)
else
    echo "=+= No NVIDIA container runtime detected - running on CPU"
fi

# Build or rebuild the Docker image before testing.
# `source` executes do_build.sh in this shell.
# do_build.sh runs `docker build`, which automatically reads
# ${SCRIPT_DIR}/Dockerfile and creates the image named segment-algorithm.
echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

# Cleanup runs whenever this script exits, whether it succeeds, fails,
# or is interrupted with Ctrl+C.
cleanup() {
    echo "=+= Cleaning permissions ..."
    docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "chmod -R -f o+rwX /output/* || true"

    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
}

chmod -R -f o+rX "$INPUT_DIR"

if [ -d "${OUTPUT_DIR}/interface_1" ]; then
  chmod -f o+rwX "${OUTPUT_DIR}/interface_1"

  echo "=+= Cleaning up any earlier output"
  docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "${OUTPUT_DIR}/interface_1":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interface_1"
fi

docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

trap cleanup EXIT

run_docker_forward_pass() {
    local interface_dir="$1"

    echo "=+= Doing a forward pass on ${interface_dir}"

    ## Note the extra arguments that are passed here:
    # '--network none'
    #    entails there is no internet connection
    # '--volume <NAME>:/tmp'
    #   is added because on the ORena FOCUS Challenge platform this directory cannot be used to store permanent files
    # '--shm-size' is left at Docker's default (64 MB). A PyTorch DataLoader with
    #   num_workers > 0 needs more shared memory — if you use one, add e.g.
    #   '--shm-size=2g' here.
    docker run --rm \
        --platform=linux/amd64 \
        --network none \
        "${GPU_FLAGS[@]}" \
        --volume "${INPUT_DIR}/${interface_dir}":/input:ro \
        --volume "${OUTPUT_DIR}/${interface_dir}":/output \
        --volume "$DOCKER_NOOP_VOLUME":/tmp \
        "$DOCKER_IMAGE_TAG"

  echo "=+= Wrote results to ${OUTPUT_DIR}/${interface_dir}"
}

run_docker_forward_pass "interface_1"

echo "=+= Save this image for uploading via ./do_save.sh"
