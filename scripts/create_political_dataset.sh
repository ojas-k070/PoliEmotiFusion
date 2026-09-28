#!/usr/bin/env bash
set -e

cd /home/parth_kahane/PoliEmotiFusion
mkdir -p data/political_emotion

for d in data/image_emotion/train/*; do
  [ -d "$d" ] || continue
  label=$(basename "$d")
  mkdir -p "data/political_emotion/$label"
  cp "$d"/*.jpg "data/political_emotion/$label/" 2>/dev/null || true
  if [ -d "data/image_emotion/val/$label" ]; then
    cp "data/image_emotion/val/$label"/*.jpg "data/political_emotion/$label/" 2>/dev/null || true
  fi
done

echo "WARNING: This smoke-test copy does not verify that the source images are political."
echo "smoke_test_data_ready"
for d in data/political_emotion/*; do
  [ -d "$d" ] || continue
  echo "$(basename "$d") $(find "$d" -type f | wc -l)"
done
