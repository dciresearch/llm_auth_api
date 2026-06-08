#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
docker run --rm -it --name main --gpus all --network host \
  --mount "src=${SCRIPT_DIR},target=/workdir,type=bind" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  docker_manager bash -c "python run_api.py"
