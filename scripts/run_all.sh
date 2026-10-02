#!/usr/bin/env bash
cd "$(dirname "$0")/.."
bash scripts/run_queue.sh && bash scripts/run_post.sh
