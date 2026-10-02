#!/usr/bin/env bash
# Stream-download each MVTec 3D-AD category archive (no archive kept on disk), convert it to compact npz
# files with scripts/prep_mvtec3d.py and delete the raw extraction. Licence: CC BY-NC-SA 4.0 (MVTec Software GmbH).
# Usage: bash scripts/download_mvtec3d.sh [category ...]
set -euo pipefail
cd "$(dirname "$0")/.."
export SSL_CERT_FILE=${SSL_CERT_FILE:-/etc/pki/tls/certs/ca-bundle.crt}
PAGE=https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad/downloads
mkdir -p data/raw data/proc
curl -sL "$PAGE" -o data/raw/downloads.html
CATS=("$@")
if [ ${#CATS[@]} -eq 0 ]; then
  CATS=(bagel cable_gland carrot cookie dowel foam peach potato rope tire)
fi
for c in "${CATS[@]}"; do
  if [ -f "data/proc/$c/prep_summary.json" ]; then echo "skip $c"; continue; fi
  url=$(grep -o -E "https://www.mydrive.ch/[^\"]*/$c\.tar\.xz" data/raw/downloads.html | head -1)
  echo "$(date -u +%T) download $c from $url"
  if [ ! -d "data/raw/$c" ]; then
    curl -sL "$url" | tar -xJ -C data/raw --wildcards "$c/*/*/rgb/*" "$c/*/*/xyz/*" "$c/*/*/gt/*" "$c/license.txt" "$c/calibration/*"
  fi
  echo "$(date -u +%T) convert $c"
  .venv/bin/python scripts/prep_mvtec3d.py "data/raw/$c" --out data/proc
  cp "data/raw/$c/license.txt" data/proc/LICENSE_MVTec3D-AD.txt
  rm -rf "data/raw/$c"
done
echo "$(date -u +%T) done"
