"""Exact ground truth on the same seeded base (rebuilt in numpy) and recall@10 of both modes."""
import sys, numpy as np
lut = np.load(sys.argv[1]); vm8 = np.load(sys.argv[2])
q = lut["q"]; assert np.array_equal(q, vm8["q"])
rng = np.random.default_rng(0); N, dim = 200_000, 768
base = np.empty((N, dim), dtype=np.float32)
for s in range(0, N, 100_000):
    base[s:s+100_000] = rng.random((100_000, dim), dtype=np.float32)   # same draws as cells_2bit.ensure_index
# inner product on normalised rows? turbovec ranks by inner product on the stored (normalised) vectors:
bn = base / np.linalg.norm(base, axis=1, keepdims=True)
qn = q / np.linalg.norm(q, axis=1, keepdims=True)
gt = np.empty((len(q), 10), dtype=np.int64)
for i in range(0, len(q), 50):
    sc = qn[i:i+50] @ bn.T
    gt[i:i+50] = np.argpartition(-sc, 10, axis=1)[:, :10]
def recall(ids):
    return np.mean([len(set(ids[i].tolist()) & set(gt[i].tolist())) / 10 for i in range(len(q))])
r_l, r_v = recall(lut["ids"]), recall(vm8["ids"])
agree = np.mean([len(set(lut["ids"][i].tolist()) & set(vm8["ids"][i].tolist())) / 10 for i in range(len(q))])
print(f"recall@10 vs exact: LUT {r_l:.4f}  vm8-SMMLA {r_v:.4f}  (delta {r_v-r_l:+.4f}); top-10 set overlap LUT vs vm8 {agree:.4f}")
