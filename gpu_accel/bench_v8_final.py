"""Final v8 benchmark: fused ttest + threshold_to_indices.

Proper A/B comparison with component-level breakdown.
"""

import time
import numpy as np
import sys
sys.path.insert(0, "/Users/sharif/Code/mne-python")

import mne
from mne.stats.cluster_level import (
    _do_1samp_permutations,
    _setup_adjacency,
    _st_fused_ccl,
    _fused_ttest,
    _threshold_to_indices,
    _threshold_to_indices_neg,
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

print(f"Setup: {n_subjects} subjects × {n_src} vertices × {n_times} times = {n_tests:,} tests")
print(f"has_numba = {has_numba}")
print("=" * 70)


class DummyPB:
    def update(self, n): pass


rng = np.random.RandomState(42)
n_p = 2048
orders = [rng.randint(0, 2, size=n_subjects).astype(bool) for _ in range(n_p)]

# ---- v8 via _do_1samp_permutations ----
print("\n--- v8 (fused ttest + threshold_to_indices) via _do_1samp_permutations ---")
# Warmup
_do_1samp_permutations(
    X, slices=None, threshold=threshold, tail=1,
    adjacency=adj_list, stat_fun=ttest_1samp_no_p,
    max_step=1, include=None, partitions=None,
    t_power=1, orders=orders[:10], sample_shape=(n_tests,),
    buffer_size=None, progress_bar=DummyPB(),
)

t0 = time.perf_counter()
result_v8 = _do_1samp_permutations(
    X, slices=None, threshold=threshold, tail=1,
    adjacency=adj_list, stat_fun=ttest_1samp_no_p,
    max_step=1, include=None, partitions=None,
    t_power=1, orders=orders, sample_shape=(n_tests,),
    buffer_size=None, progress_bar=DummyPB(),
)
t1 = time.perf_counter()
time_v8 = t1 - t0
print(f"  {n_p} perms: {time_v8:.3f}s  ({time_v8/n_p*1e3:.3f} ms/perm)")

# Linear regression
times_v8 = []
perms_v8 = []
for n_pp in [256, 512, 1024, 2048]:
    ords = [rng.randint(0, 2, size=n_subjects).astype(bool) for _ in range(n_pp)]
    t0 = time.perf_counter()
    _do_1samp_permutations(
        X, slices=None, threshold=threshold, tail=1,
        adjacency=adj_list, stat_fun=ttest_1samp_no_p,
        max_step=1, include=None, partitions=None,
        t_power=1, orders=ords, sample_shape=(n_tests,),
        buffer_size=None, progress_bar=DummyPB(),
    )
    t1 = time.perf_counter()
    times_v8.append(t1 - t0)
    perms_v8.append(n_pp)
A = np.vstack([perms_v8, np.ones(len(perms_v8))]).T
slope_v8, intercept_v8 = np.linalg.lstsq(A, times_v8, rcond=None)[0]
print(f"  Linear: overhead={intercept_v8*1e3:.1f}ms, per_perm={slope_v8*1e3:.3f}ms")


# ---- v7 manual inline (numpy ttest, same CCL) ----
print("\n--- v7 (numpy ttest) manual inline ---")
n_samp, n_vars = X.shape
_sum_sq = np.sum(X**2, axis=0)
_sqrt_n_nm1 = np.sqrt(n_samp * (n_samp - 1))
_mean_s = np.empty(n_vars, dtype=np.float64)
_denom_sq = np.empty(n_vars, dtype=np.float64)
_t_buf = np.empty(n_vars, dtype=np.float64)

# ST setup
_st_n_src = len(adj_list)
_st_lengths = np.array([len(a) for a in adj_list])
_st_indptr = np.zeros(_st_n_src + 1, dtype=np.intp)
np.cumsum(_st_lengths, out=_st_indptr[1:])
_st_indices = np.concatenate(adj_list).astype(np.intp)
_st_flat_map = -np.ones(n_vars, dtype=np.intp)
_dummy = np.array([0], dtype=np.intp)
_st_fused_ccl(_dummy, 1, _st_flat_map, _st_indptr, _st_indices, _st_n_src, 1)

# Warmup
for order in orders[:10]:
    signs_1d = 2.0 * order - 1.0
    np.dot(signs_1d, X, out=_mean_s)

max_sums_v7 = np.empty(n_p, dtype=np.double)
t0 = time.perf_counter()
for seed_idx, order in enumerate(orders):
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

    _st_best = 0.0
    x_in = _t_buf > threshold
    act = np.where(x_in)[0].astype(np.intp)
    n_act = len(act)
    if n_act > 0:
        comps = _st_fused_ccl(
            act, n_act, _st_flat_map,
            _st_indptr, _st_indices, _st_n_src, 1,
        )
        sums = np.bincount(comps, weights=_t_buf[act])
        idx = np.argmax(np.abs(sums))
        if abs(sums[idx]) > abs(_st_best):
            _st_best = sums[idx]
    max_sums_v7[seed_idx] = _st_best
t1 = time.perf_counter()
time_v7 = t1 - t0
print(f"  {n_p} perms: {time_v7:.3f}s  ({time_v7/n_p*1e3:.3f} ms/perm)")

# Parity
print(f"\n  Results match: {np.allclose(result_v8, max_sums_v7, atol=1e-10)}")

# ---- Overall speedup ----
print(f"\n{'='*70}")
print(f"v8: {time_v8/n_p*1e3:.3f} ms/perm (linear: {slope_v8*1e3:.3f})")
print(f"v7: {time_v7/n_p*1e3:.3f} ms/perm")
print(f"Speedup: {time_v7/time_v8:.2f}x")


# ---- Per-component breakdown (1024 perms) ----
print(f"\n{'='*70}")
print("Per-component breakdown (1024 perms)")
orders_1k = orders[:1024]
n = len(orders_1k)

# v8 breakdown
X_T = np.ascontiguousarray(X.T)
inv_n = 1.0 / n_samp
neg_n = -float(n_samp)
t_buf2 = np.empty(n_vars, dtype=np.float64)
idx_buf = np.empty(n_vars, dtype=np.intp)
_fused_ttest(X_T, np.ones(n_samp), _sum_sq, inv_n, neg_n, _sqrt_n_nm1, t_buf2)
_threshold_to_indices(t_buf2, 0.0, idx_buf)

t_signs = t_fused = t_thresh_idx = t_ccl = t_bincount = 0.0

for order in orders_1k:
    t0 = time.perf_counter()
    signs_1d = 2.0 * order - 1.0
    t1 = time.perf_counter()
    _fused_ttest(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1, t_buf2)
    t2 = time.perf_counter()
    n_act = _threshold_to_indices(t_buf2, threshold, idx_buf)
    act = idx_buf[:n_act]
    t3 = time.perf_counter()
    if n_act > 0:
        comps = _st_fused_ccl(
            act, n_act, _st_flat_map,
            _st_indptr, _st_indices, _st_n_src, 1,
        )
        t4 = time.perf_counter()
        sums = np.bincount(comps, weights=t_buf2[act])
        t5 = time.perf_counter()
    else:
        t4 = t3
        t5 = t3
    t_signs += t1 - t0
    t_fused += t2 - t1
    t_thresh_idx += t3 - t2
    t_ccl += t4 - t3
    t_bincount += t5 - t4

total_v8b = t_signs + t_fused + t_thresh_idx + t_ccl + t_bincount
print(f"\nv8 (fused ttest + _threshold_to_indices):")
print(f"  signs:        {t_signs/n*1e6:7.1f} µs  ({t_signs/total_v8b*100:5.1f}%)")
print(f"  fused ttest:  {t_fused/n*1e6:7.1f} µs  ({t_fused/total_v8b*100:5.1f}%)")
print(f"  thresh+idx:   {t_thresh_idx/n*1e6:7.1f} µs  ({t_thresh_idx/total_v8b*100:5.1f}%)")
print(f"  CCL (UF):     {t_ccl/n*1e6:7.1f} µs  ({t_ccl/total_v8b*100:5.1f}%)")
print(f"  bincount:     {t_bincount/n*1e6:7.1f} µs  ({t_bincount/total_v8b*100:5.1f}%)")
print(f"  TOTAL:        {total_v8b/n*1e6:7.1f} µs")

# v7 breakdown
t_signs7 = t_dot7 = t_elem7 = t_tw7 = t_ccl7 = t_bincount7 = 0.0

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
    act = np.where(x_in)[0].astype(np.intp)
    n_act = len(act)
    t4 = time.perf_counter()
    if n_act > 0:
        comps = _st_fused_ccl(
            act, n_act, _st_flat_map,
            _st_indptr, _st_indices, _st_n_src, 1,
        )
        t5 = time.perf_counter()
        sums = np.bincount(comps, weights=_t_buf[act])
        t6 = time.perf_counter()
    else:
        t5 = t4
        t6 = t4
    t_signs7 += t1 - t0
    t_dot7 += t2 - t1
    t_elem7 += t3 - t2
    t_tw7 += t4 - t3
    t_ccl7 += t5 - t4
    t_bincount7 += t6 - t5

total_v7b = t_signs7 + t_dot7 + t_elem7 + t_tw7 + t_ccl7 + t_bincount7
print(f"\nv7 (numpy ttest + np.where):")
print(f"  signs:        {t_signs7/n*1e6:7.1f} µs  ({t_signs7/total_v7b*100:5.1f}%)")
print(f"  dot:          {t_dot7/n*1e6:7.1f} µs  ({t_dot7/total_v7b*100:5.1f}%)")
print(f"  elementwise:  {t_elem7/n*1e6:7.1f} µs  ({t_elem7/total_v7b*100:5.1f}%)")
print(f"  thresh+where: {t_tw7/n*1e6:7.1f} µs  ({t_tw7/total_v7b*100:5.1f}%)")
print(f"  CCL (UF):     {t_ccl7/n*1e6:7.1f} µs  ({t_ccl7/total_v7b*100:5.1f}%)")
print(f"  bincount:     {t_bincount7/n*1e6:7.1f} µs  ({t_bincount7/total_v7b*100:5.1f}%)")
print(f"  TOTAL:        {total_v7b/n*1e6:7.1f} µs")

print(f"\nTtest component:     {(t_dot7+t_elem7)/n*1e6:.1f} µs (v7) → {t_fused/n*1e6:.1f} µs (v8) = {(t_dot7+t_elem7)/t_fused:.2f}x")
print(f"Thresh+indices:      {t_tw7/n*1e6:.1f} µs (v7) → {t_thresh_idx/n*1e6:.1f} µs (v8) = {t_tw7/t_thresh_idx:.2f}x")
print(f"Total:               {total_v7b/n*1e6:.1f} µs (v7) → {total_v8b/n*1e6:.1f} µs (v8) = {total_v7b/total_v8b:.2f}x")
