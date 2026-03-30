"""Proper A/B: toggle between fused and numpy ttest within same code path.

Tests both on the exact same data/orders for a fair comparison.
"""

import time
import os
import numpy as np
import sys
sys.path.insert(0, "/Users/sharif/Code/mne-python")

import mne
from mne.stats.cluster_level import (
    _do_1samp_permutations,
    _setup_adjacency,
    _st_fused_ccl,
    _fused_ttest,
)
from mne.stats.parametric import ttest_1samp_no_p
from mne.fixes import has_numba

# ---- Setup ----
data_path = mne.datasets.sample.data_path()
subjects_dir = data_path / "subjects"
src = mne.read_source_spaces(
    subjects_dir / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif", verbose=False
)
adjacency = mne.spatial_src_adjacency(src, verbose=False)
n_src = adjacency.shape[0]
n_subjects = 7
n_times = 15
n_tests = n_src * n_times

np.random.seed(123)
X = np.random.randn(n_subjects, n_tests) * 0.3
signal_verts = np.arange(100, 200)
for t in range(5, 10):
    X[:, signal_verts + t * n_src] += 2.0

adj_list = _setup_adjacency(adjacency, n_tests, n_times)
threshold = 1.67
max_step = 1

print(f"Setup: {n_subjects} subjects × {n_src} vertices × {n_times} times = {n_tests:,} tests")
print(f"has_numba = {has_numba}")
print("=" * 70)


class DummyPB:
    def update(self, n): pass


# Generate same orders for both paths
rng = np.random.RandomState(42)
n_p = 2048
orders = [rng.randint(0, 2, size=n_subjects).astype(bool) for _ in range(n_p)]

# ---- v8 (fused ttest) ----
# Warmup
_do_1samp_permutations(
    X, slices=None, threshold=threshold, tail=1,
    adjacency=adj_list, stat_fun=ttest_1samp_no_p,
    max_step=max_step, include=None, partitions=None,
    t_power=1, orders=orders[:10], sample_shape=(n_tests,),
    buffer_size=None, progress_bar=DummyPB(),
)

t0 = time.perf_counter()
result_v8 = _do_1samp_permutations(
    X, slices=None, threshold=threshold, tail=1,
    adjacency=adj_list, stat_fun=ttest_1samp_no_p,
    max_step=max_step, include=None, partitions=None,
    t_power=1, orders=orders, sample_shape=(n_tests,),
    buffer_size=None, progress_bar=DummyPB(),
)
t1 = time.perf_counter()
time_v8 = t1 - t0
per_perm_v8 = time_v8 / n_p * 1000
print(f"v8 (fused ttest):  {time_v8:.3f}s  ({per_perm_v8:.3f} ms/perm)")


# ---- Manually run the numpy ttest path (inline, same loop structure) ----
# This replicates the exact v7 inner loop logic without modifying cluster_level.py

# Setup (same as _do_1samp_permutations would do)
n_samp, n_vars = X.shape
_sum_sq = np.sum(X**2, axis=0)
_sqrt_n_nm1 = np.sqrt(n_samp * (n_samp - 1))
_mean_s = np.empty(n_vars, dtype=np.float64)
_denom_sq = np.empty(n_vars, dtype=np.float64)
_t_buf = np.empty(n_vars, dtype=np.float64)

# ST fast path setup
_st_n_src = len(adj_list)
_st_n_times = n_vars // _st_n_src
_st_lengths = np.array([len(a) for a in adj_list])
_st_indptr = np.zeros(_st_n_src + 1, dtype=np.intp)
np.cumsum(_st_lengths, out=_st_indptr[1:])
_st_indices = np.concatenate(adj_list).astype(np.intp)
_st_flat_map = -np.ones(n_vars, dtype=np.intp)
_dummy_idx = np.array([0], dtype=np.intp)
_st_fused_ccl(_dummy_idx, 1, _st_flat_map, _st_indptr, _st_indices, _st_n_src, max_step)

max_cluster_sums_v7 = np.empty(len(orders), dtype=np.double)

# Warmup
for order in orders[:10]:
    signs_1d = 2.0 * order - 1.0
    np.dot(signs_1d, X, out=_mean_s)
    _mean_s /= n_samp
    np.multiply(_mean_s, _mean_s, out=_denom_sq)
    _denom_sq *= -n_samp
    _denom_sq += _sum_sq
    np.maximum(_denom_sq, 0, out=_denom_sq)
    np.sqrt(_denom_sq, out=_denom_sq)
    np.divide(_mean_s, _denom_sq, out=_t_buf)
    _t_buf *= _sqrt_n_nm1
    x_in = _t_buf > threshold
    act_idx = np.where(x_in)[0].astype(np.intp)

t0 = time.perf_counter()
for seed_idx, order in enumerate(orders):
    signs_1d = 2.0 * order - 1.0
    # v7 numpy ttest
    np.dot(signs_1d, X, out=_mean_s)
    _mean_s /= n_samp
    np.multiply(_mean_s, _mean_s, out=_denom_sq)
    _denom_sq *= -n_samp
    _denom_sq += _sum_sq
    np.maximum(_denom_sq, 0, out=_denom_sq)
    np.sqrt(_denom_sq, out=_denom_sq)
    np.divide(_mean_s, _denom_sq, out=_t_buf)
    _t_buf *= _sqrt_n_nm1

    # same CCL path as v8
    _st_best = 0.0
    x_in = _t_buf > threshold
    _st_act_idx = np.where(x_in)[0].astype(np.intp)
    _st_n_act = len(_st_act_idx)
    if _st_n_act > 0:
        comps = _st_fused_ccl(
            _st_act_idx, _st_n_act, _st_flat_map,
            _st_indptr, _st_indices, _st_n_src, max_step,
        )
        sums = np.bincount(comps, weights=_t_buf[_st_act_idx])
        _st_idx = np.argmax(np.abs(sums))
        if abs(sums[_st_idx]) > abs(_st_best):
            _st_best = sums[_st_idx]
    max_cluster_sums_v7[seed_idx] = _st_best

t1 = time.perf_counter()
time_v7 = t1 - t0
per_perm_v7 = time_v7 / n_p * 1000
print(f"v7 (numpy ttest):  {time_v7:.3f}s  ({per_perm_v7:.3f} ms/perm)")

# ---- Summary ----
print(f"\nSpeedup: {per_perm_v7/per_perm_v8:.2f}x")
print(f"Savings: {per_perm_v7 - per_perm_v8:.3f} ms/perm")

# Parity check
print(f"\nResults match: {np.allclose(result_v8, max_cluster_sums_v7, atol=1e-10)}")
if not np.allclose(result_v8, max_cluster_sums_v7, atol=1e-10):
    diffs = np.abs(result_v8 - max_cluster_sums_v7)
    print(f"  Max diff: {diffs.max():.2e}")
    print(f"  Mismatches: {np.sum(diffs > 1e-10)}")


# ---- Per-component breakdown in actual loop ----
print("\n--- Per-component breakdown (v8 fused, 1024 perms) ---")
orders_1k = orders[:1024]

# Pre-setup (same as _do_1samp_permutations)
X_T = np.ascontiguousarray(X.T)
inv_n = 1.0 / n_samp
neg_n = -float(n_samp)
t_buf2 = np.empty(n_vars, dtype=np.float64)
# Warmup
_fused_ttest(X_T, np.ones(n_samp), _sum_sq, inv_n, neg_n, _sqrt_n_nm1, t_buf2)

t_signs = t_fused = t_thresh = t_where = t_ccl = t_bincount = 0.0

for order in orders_1k:
    t0 = time.perf_counter()
    signs_1d = 2.0 * order - 1.0
    t1 = time.perf_counter()
    _fused_ttest(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1, t_buf2)
    t2 = time.perf_counter()
    x_in = t_buf2 > threshold
    t3 = time.perf_counter()
    act_idx = np.where(x_in)[0].astype(np.intp)
    t4 = time.perf_counter()
    n_act = len(act_idx)
    if n_act > 0:
        comps = _st_fused_ccl(
            act_idx, n_act, _st_flat_map,
            _st_indptr, _st_indices, _st_n_src, max_step,
        )
        t5 = time.perf_counter()
        sums = np.bincount(comps, weights=t_buf2[act_idx])
        t6 = time.perf_counter()
    else:
        t5 = t4
        t6 = t4

    t_signs += t1 - t0
    t_fused += t2 - t1
    t_thresh += t3 - t2
    t_where += t4 - t3
    t_ccl += t5 - t4
    t_bincount += t6 - t5

total = t_signs + t_fused + t_thresh + t_where + t_ccl + t_bincount
n = len(orders_1k)
print(f"  signs:      {t_signs/n*1e6:7.1f} µs  ({t_signs/total*100:5.1f}%)")
print(f"  fused ttest:{t_fused/n*1e6:7.1f} µs  ({t_fused/total*100:5.1f}%)")
print(f"  threshold:  {t_thresh/n*1e6:7.1f} µs  ({t_thresh/total*100:5.1f}%)")
print(f"  np.where:   {t_where/n*1e6:7.1f} µs  ({t_where/total*100:5.1f}%)")
print(f"  CCL (UF):   {t_ccl/n*1e6:7.1f} µs  ({t_ccl/total*100:5.1f}%)")
print(f"  bincount:   {t_bincount/n*1e6:7.1f} µs  ({t_bincount/total*100:5.1f}%)")
print(f"  TOTAL:      {total/n*1e6:7.1f} µs")

# Compare with v7 breakdown
print("\n--- Per-component breakdown (v7 numpy, 1024 perms) ---")
t_signs7 = t_dot7 = t_elem7 = t_thresh7 = t_where7 = t_ccl7 = t_bincount7 = 0.0

for order in orders_1k:
    t0 = time.perf_counter()
    signs_1d = 2.0 * order - 1.0
    t1 = time.perf_counter()
    np.dot(signs_1d, X, out=_mean_s)
    t2 = time.perf_counter()
    _mean_s /= n_samp
    np.multiply(_mean_s, _mean_s, out=_denom_sq)
    _denom_sq *= -n_samp
    _denom_sq += _sum_sq
    np.maximum(_denom_sq, 0, out=_denom_sq)
    np.sqrt(_denom_sq, out=_denom_sq)
    np.divide(_mean_s, _denom_sq, out=_t_buf)
    _t_buf *= _sqrt_n_nm1
    t3 = time.perf_counter()
    x_in = _t_buf > threshold
    t4 = time.perf_counter()
    act_idx = np.where(x_in)[0].astype(np.intp)
    t5 = time.perf_counter()
    n_act = len(act_idx)
    if n_act > 0:
        comps = _st_fused_ccl(
            act_idx, n_act, _st_flat_map,
            _st_indptr, _st_indices, _st_n_src, max_step,
        )
        t6 = time.perf_counter()
        sums = np.bincount(comps, weights=_t_buf[act_idx])
        t7 = time.perf_counter()
    else:
        t6 = t5
        t7 = t5

    t_signs7 += t1 - t0
    t_dot7 += t2 - t1
    t_elem7 += t3 - t2
    t_thresh7 += t4 - t3
    t_where7 += t5 - t4
    t_ccl7 += t6 - t5
    t_bincount7 += t7 - t6

total7 = t_signs7 + t_dot7 + t_elem7 + t_thresh7 + t_where7 + t_ccl7 + t_bincount7
print(f"  signs:      {t_signs7/n*1e6:7.1f} µs  ({t_signs7/total7*100:5.1f}%)")
print(f"  dot:        {t_dot7/n*1e6:7.1f} µs  ({t_dot7/total7*100:5.1f}%)")
print(f"  elementwise:{t_elem7/n*1e6:7.1f} µs  ({t_elem7/total7*100:5.1f}%)")
print(f"  threshold:  {t_thresh7/n*1e6:7.1f} µs  ({t_thresh7/total7*100:5.1f}%)")
print(f"  np.where:   {t_where7/n*1e6:7.1f} µs  ({t_where7/total7*100:5.1f}%)")
print(f"  CCL (UF):   {t_ccl7/n*1e6:7.1f} µs  ({t_ccl7/total7*100:5.1f}%)")
print(f"  bincount:   {t_bincount7/n*1e6:7.1f} µs  ({t_bincount7/total7*100:5.1f}%)")
print(f"  TOTAL:      {total7/n*1e6:7.1f} µs")

print(f"\nTtest component: {(t_dot7+t_elem7)/n*1e6:.1f} µs (v7) → {t_fused/n*1e6:.1f} µs (v8) = "
      f"{(t_dot7+t_elem7)/t_fused:.2f}x")
print(f"Total:          {total7/n*1e6:.1f} µs (v7) → {total/n*1e6:.1f} µs (v8) = "
      f"{total7/total:.2f}x")
