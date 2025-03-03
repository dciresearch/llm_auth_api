#!/bin/bash

nvcc --version
nvidia-smi
python -u -m vllm.entrypoints.openai.api_server "$@"