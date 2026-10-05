#!/bin/bash
# Like smoke.sh, but one .so and an env toggle per label: A = toggle off, B = TURBOVEC_2BIT_VM8=1.
set -uo pipefail
source ~/venv/bin/activate
ARCH=$1; SO=$2; CELLS=${3:-nq100_st}
if [ "$ARCH" = x86 ]; then export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libopenblas.so.0; else export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libopenblas.so.0; fi
DEST=~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so
IDX=$HOME/.cache/turbovec-hillclimb/cells_200000_2bit.tvim
cp ~/hc/so/${SO}.so $DEST
cd ~/hc
for tag in lut vm8 vm8 lut; do
  if [ "$tag" = vm8 ]; then export TURBOVEC_2BIT_VM8=1; else unset TURBOVEC_2BIT_VM8; fi
  for cell in $CELLS; do
    case $cell in
      nq1_st)   nq=1;   th=1;;
      nq1_mt)   nq=1;   th=$(nproc);;
      nq100_st) nq=100; th=1;;
      *)        nq=100; th=$(nproc);;
    esac
    ms=$(RAYON_NUM_THREADS=$th python -c "
import time,numpy as np
from turbovec import IdMapIndex
idx=IdMapIndex.load('$IDX')
q=np.random.default_rng(7).random(($nq,768),dtype=np.float32)
idx.search(q,k=10); best=1e9
for _ in range(30 if $nq==100 else 200):
    t0=time.perf_counter(); idx.search(q,k=10); best=min(best,time.perf_counter()-t0)
print(round(best*1e3,3))" 2>/dev/null)
    echo "$tag $cell $ms"
  done
done
echo SMOKE_DONE
