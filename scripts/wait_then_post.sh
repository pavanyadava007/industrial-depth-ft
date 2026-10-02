#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q "queue finished" logs/queue.log; do sleep 30; done
bash scripts/run_post.sh
