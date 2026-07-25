#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DOCKER_IMAGE_TAG="segment-algorithm"

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

build_timestamp=$( docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")

if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for container $DOCKER_IMAGE_TAG"
    exit 1
fi

formatted_build_info=$(echo $build_timestamp | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')

output_filename="${SCRIPT_DIR}/${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"

echo "==+=="
echo "Saving the container image as ${output_filename}. This can take a while."
echo ""

# Use pigz (parallel gzip) if available — much faster than gzip on a large image.
if command -v pigz >/dev/null 2>&1; then
    compressor=(pigz -c)
else
    compressor=(gzip -c)
fi

# Show a progress bar with pv if available, sized by the image's uncompressed size.
image_size=$(docker image inspect "$DOCKER_IMAGE_TAG" --format '{{.Size}}' 2>/dev/null || true)

if command -v pv >/dev/null 2>&1; then
    if [[ "$image_size" =~ ^[0-9]+$ && "$image_size" -gt 0 ]]; then
        docker save "$DOCKER_IMAGE_TAG" \
            | pv --size "$image_size" --progress --timer --eta --rate --bytes \
            | "${compressor[@]}" > "$output_filename"
    else
        docker save "$DOCKER_IMAGE_TAG" | pv --timer --rate --bytes | "${compressor[@]}" > "$output_filename"
    fi
else
    echo "Tip: install 'pv' (e.g. sudo apt-get install pv) for a progress bar."
    docker save "$DOCKER_IMAGE_TAG" | "${compressor[@]}" > "$output_filename"
fi
echo "Container image saved as ${output_filename}"
echo "==+=="
