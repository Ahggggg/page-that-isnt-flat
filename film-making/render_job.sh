#!/bin/bash
# The film's three Blender shots on one cloud GPU (we used one RTX 4090: 77 minutes of machine time, about USD 1). Blender 4.5.14 LTS,
# verified by SHA-256. Phase 1: one timed frame per shot (bench.log). Phase 2: the full shots. Progress in render.log.
# Expects this repository at /workspace/page-that-isnt-flat with film-making/blender/data and film-making/app/public/data
# already built (FILM.md, steps 2-3); frames land in /workspace/out/<shot>/.
set -u
V=/workspace/page-that-isnt-flat/film-making
cd /workspace
if [ ! -x /workspace/blender/blender ]; then
  apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq libxi6 libxrender1 libxkbcommon0 libsm6 libgl1 libxxf86vm1 libxfixes3 libegl1 libglu1-mesa >/dev/null 2>&1
  curl -sSLo blender.tar.xz https://download.blender.org/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz
  echo "9ba871ff2ecd36526b77432745980b7e6664ecd0c7ca11c48849073dcfe06da3  blender.tar.xz" | sha256sum -c - || { echo BLENDER_HASH_FAIL; touch /workspace/job.done; exit 1; }
  tar -xf blender.tar.xz && mv blender-4.5.14-linux-x64 blender && rm blender.tar.xz
fi
export BLENDER_DEVICE=OPTIX
B=/workspace/blender/blender
SAMPLES_SCROLL=${SAMPLES_SCROLL:-256}; SAMPLES_RELIEF=${SAMPLES_RELIEF:-256}; SAMPLES_UNROLL=${SAMPLES_UNROLL:-256}
echo "phase bench $(date +%T)" >> render.log
for spec in "scroll 300 $SAMPLES_SCROLL" "relief 220 $SAMPLES_RELIEF" "unroll 200 $SAMPLES_UNROLL"; do
  set -- $spec
  t0=$(date +%s)
  BLENDER_SAMPLES=$3 $B -b --factory-startup -P $V/blender/$1.py -- --out /workspace/bench/$1 --still $2 > /workspace/bench-$1.out 2>&1
  echo "{\"bench\": \"$1\", \"frame\": $2, \"samples\": $3, \"seconds\": $(( $(date +%s) - t0 )), \"ok\": $( [ -f /workspace/bench/$1/still_$(printf %04d $2).png ] && echo true || echo false )}" >> bench.log
done
# wait (up to 20 min) for an optional /workspace/go file that sets sample counts sized to your budget from bench.log
for i in $(seq 1 80); do [ -f /workspace/go ] && break; sleep 15; done
[ -f /workspace/go ] && . /workspace/go
echo "phase render $(date +%T) samples scroll=$SAMPLES_SCROLL relief=$SAMPLES_RELIEF unroll=$SAMPLES_UNROLL" >> render.log
for spec in "scroll 1 360 $SAMPLES_SCROLL" "relief 1 360 $SAMPLES_RELIEF" "unroll 1 300 $SAMPLES_UNROLL"; do
  set -- $spec
  mkdir -p /workspace/out/$1
  BLENDER_SAMPLES=$4 $B -b --factory-startup -P $V/blender/$1.py -- --out /workspace/out/$1 --frames $2 $3 > /workspace/render-$1.out 2>&1
  echo "{\"shot\": \"$1\", \"exit\": $?, \"frames\": $(ls /workspace/out/$1 | wc -l), \"t\": \"$(date +%T)\"}" >> render.log
done
touch /workspace/job.done
