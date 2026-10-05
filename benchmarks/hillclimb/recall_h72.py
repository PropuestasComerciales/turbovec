"""H72 recall gate: LUT path vs vm8 SMMLA path vs exact, on the harness index (N=200k, dim 768, 2-bit).
Run twice: without and with TURBOVEC_2BIT_VM8=1 (the layout is chosen at load). Writes ids to a file per mode."""
import os, sys, json, time, numpy as np
from turbovec import IdMapIndex
mode = "vm8" if os.environ.get("TURBOVEC_2BIT_VM8") == "1" else "lut"
cache = os.environ.get("TURBOVEC_HILLCLIMB_CACHE", os.path.expanduser("~/.cache/turbovec-hillclimb"))
path = os.path.join(cache, "cells_200000_2bit.tvim")
out = sys.argv[1] if len(sys.argv) > 1 else f"/tmp/h72_{mode}.npz"
idx = IdMapIndex.load(path)
nq, k = 500, 10
q = np.random.default_rng(7).random((nq, 768), dtype=np.float32)
t0 = time.perf_counter(); res = idx.search(q, k=k); t = time.perf_counter() - t0
ids = np.asarray(res[1] if isinstance(res, tuple) else res.ids if hasattr(res, "ids") else res)
np.savez(out, ids=ids, q=q)
print(f"{mode}: nq={nq} k={k} search {t*1e3:.1f} ms -> {out}; ids shape {ids.shape}")
