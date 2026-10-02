#!/usr/bin/env bash
cd "$(dirname "$0")/.."
bash scripts/download_mvtec3d.sh cookie && bash scripts/run_queue.sh
