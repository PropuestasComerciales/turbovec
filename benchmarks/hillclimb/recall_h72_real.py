"""H72 real-data recall gate, mirroring benchmarks/suite/recall_d1536_4bit.py at 2 bits.
OpenAI d=1536 (dbpedia), 100k database / 1000 queries, seed 42, recall@1 at k for TQ and calibrated TQ+.
Run with and without TURBOVEC_2BIT_VM8=1 (same code, layout chosen at load/build)."""
import os, sys, time, numpy as np
from turbovec import TurboQuantIndex
mode = "vm8" if os.environ.get("TURBOVEC_2BIT_VM8") == "1" else "lut"
DATA_DIR = os.path.expanduser("~/data/py-turboquant"); SEED, K = 42, 64
FILE = sys.argv[1] if len(sys.argv) > 1 else "openai-1536.npy"; NDB = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
K_VALUES = [1, 2, 4, 8, 10, 16, 32, 64]
all_vecs = np.load(os.path.join(DATA_DIR, FILE)); DIM = all_vecs.shape[1]
rng = np.random.RandomState(SEED); idx = rng.permutation(len(all_vecs))
database = all_vecs[idx[:NDB]].astype(np.float32); queries = all_vecs[idx[NDB:NDB+1000]].astype(np.float32)
database /= np.linalg.norm(database, axis=-1, keepdims=True); queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
true_top1 = np.argmax(queries @ database.T, axis=1)
def r1k(pred, k): return float(np.mean([true_top1[i] in pred[i, :k] for i in range(len(true_top1))]))
out = {}
for name, calib in (("TQ", False), ("TQ+", True)):
    index = TurboQuantIndex(DIM, bit_width=2)
    if calib:
        crng = np.random.RandomState(SEED); index.calibrate(database[crng.choice(len(database), 1024, replace=False)])
    index.add(database)
    t0 = time.perf_counter(); _, ids = index.search(queries, k=K); dt = time.perf_counter() - t0
    ids = np.array(ids); out[name] = {k: round(r1k(ids, k), 4) for k in K_VALUES}
    print(f"{mode} {FILE} db={NDB} {name}: search(1000q,k=64) {dt*1e3:.0f} ms  recall@1@k " + " ".join(f"{k}:{out[name][k]:.4f}" for k in K_VALUES), flush=True)
