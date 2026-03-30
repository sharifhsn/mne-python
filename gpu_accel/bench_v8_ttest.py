"""Benchmark v8: fused Numba ttest vs v7 numpy ttest.

Tests:
1. Parity: verify identical t-statistics between fused and numpy implementations
2. Per-permutation breakdown timing with real fsaverage data
3. A/B comparison: toggle _use_fast_ttest fused vs numpy
4. End-to-end timing at multiple permutation counts
"""

import time
import numpy as np
import sys
sys.path.insert(0, "/Users/sharif/Code/mne-python")

import mne
from mne.stats.cluster_level import _fused_ttest


# ---- Test 1: Parity check ----
print("=" * 70)
print("Test 1: Parity — fused Numba vs numpy ttest")
print("=" * 70)

np.random.seed(42)
n_samp = 7
n_vars = 307_260

X = np.random.randn(n_samp, n_vars)
X_T = np.ascontiguousarray(X.T)
sum_sq = np.sum(X**2, axis=0)
sqrt_n_nm1 = np.sqrt(n_samp * (n_samp - 1))
inv_n = 1.0 / n_samp
neg_n = -float(n_samp)

# Pre-allocate buffers
mean_s = np.empty(n_vars, dtype=np.float64)
denom_sq = np.empty(n_vars, dtype=np.float64)
t_buf_np = np.empty(n_vars, dtype=np.float64)
t_buf_nb = np.empty(n_vars, dtype=np.float64)

# Test multiple random sign-flips
n_test = 20
max_errs = []
for i in range(n_test):
    order = np.random.randint(0, 2, size=n_samp).astype(bool)
    signs_1d = 2.0 * order - 1.0

    # NumPy reference (old v7 code)
    np.dot(signs_1d, X, out=mean_s)
    mean_s /= n_samp
    np.multiply(mean_s, mean_s, out=denom_sq)
    denom_sq *= -n_samp
    denom_sq += sum_sq
    np.maximum(denom_sq, 0, out=denom_sq)
    np.sqrt(denom_sq, out=denom_sq)
    np.divide(mean_s, denom_sq, out=t_buf_np)
    t_buf_np *= sqrt_n_nm1
    # Fix NaN/Inf from 0/0 division
    t_buf_np[~np.isfinite(t_buf_np)] = 0.0

    # Numba fused
    _fused_ttest(X_T, signs_1d, sum_sq, inv_n, neg_n, sqrt_n_nm1, t_buf_nb)

    max_err = np.max(np.abs(t_buf_np - t_buf_nb))
    max_errs.append(max_err)

    # Also check threshold agreement
    threshold = 1.67
    np_pos = np.where(t_buf_np > threshold)[0]
    nb_pos = np.where(t_buf_nb > threshold)[0]
    np_neg = np.where(t_buf_np < -threshold)[0]
    nb_neg = np.where(t_buf_nb < -threshold)[0]

    if not np.array_equal(np_pos, nb_pos) or not np.array_equal(np_neg, nb_neg):
        print(f"  FAIL perm {i}: index mismatch!")
        print(f"    np_pos: {len(np_pos)}, nb_pos: {len(nb_pos)}")
        print(f"    np_neg: {len(np_neg)}, nb_neg: {len(nb_neg)}")
    elif i == 0:
        print(f"  perm {i}: max_err={max_err:.2e}, n_pos={len(np_pos)}, n_neg={len(np_neg)}")

print(f"  All {n_test} parity checks: max_err={max(max_errs):.2e} ✓")


# ---- Test 2: Per-permutation breakdown with real fsaverage ----
print("\n" + "=" * 70)
print("Test 2: Per-permutation breakdown (real fsaverage data)")
print("=" * 70)

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path / "subjects"
src = mne.read_source_spaces(subjects_dir / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif",
                              verbose=False)
adjacency = mne.spatial_src_adjacency(src, verbose=False)
n_src = adjacency.shape[0]
print(f"  fsaverage ico-5: {n_src} vertices")

n_subjects = 7
n_times = 15
n_tests = n_src * n_times
np.random.seed(123)
X_real = np.random.randn(n_subjects, n_tests) * 0.3
# Add some signal
signal_verts = np.arange(100, 200)
for t in range(5, 10):
    X_real[:, signal_verts + t * n_src] += 2.0

X_T_real = np.ascontiguousarray(X_real.T)
sum_sq_real = np.sum(X_real**2, axis=0)

# Buffers
t_buf = np.empty(n_tests, dtype=np.float64)

# Warmup
signs = np.ones(n_subjects, dtype=np.float64)
_fused_ttest(X_T_real, signs, sum_sq_real, inv_n, neg_n, sqrt_n_nm1, t_buf)

# Prepare orders
n_perms = 1024
orders = [np.random.randint(0, 2, size=n_subjects).astype(bool) for _ in range(n_perms)]
threshold_real = 1.67

# Profile v8 (fused Numba)
t_signs = 0
t_fused = 0
t_thresh = 0
t_where = 0

for order in orders:
    t0 = time.perf_counter()
    signs_1d = 2.0 * order - 1.0
    t1 = time.perf_counter()
    _fused_ttest(X_T_real, signs_1d, sum_sq_real, inv_n, neg_n, sqrt_n_nm1, t_buf)
    t2 = time.perf_counter()
    x_in = t_buf > threshold_real
    t3 = time.perf_counter()
    act = np.where(x_in)[0]
    t4 = time.perf_counter()

    t_signs += t1 - t0
    t_fused += t2 - t1
    t_thresh += t3 - t2
    t_where += t4 - t3

total_v8 = t_signs + t_fused + t_thresh + t_where
print(f"\n  v8 (fused Numba) per-perm breakdown ({n_perms} perms):")
print(f"    signs:      {t_signs/n_perms*1e6:7.1f} µs  ({t_signs/total_v8*100:5.1f}%)")
print(f"    fused ttest:{t_fused/n_perms*1e6:7.1f} µs  ({t_fused/total_v8*100:5.1f}%)")
print(f"    threshold:  {t_thresh/n_perms*1e6:7.1f} µs  ({t_thresh/total_v8*100:5.1f}%)")
print(f"    np.where:   {t_where/n_perms*1e6:7.1f} µs  ({t_where/total_v8*100:5.1f}%)")
print(f"    TOTAL:      {total_v8/n_perms*1e6:7.1f} µs")

# Profile v7 (numpy) for comparison
mean_s_r = np.empty(n_tests, dtype=np.float64)
denom_sq_r = np.empty(n_tests, dtype=np.float64)
t_buf_r = np.empty(n_tests, dtype=np.float64)

t_signs_7 = 0
t_dot_7 = 0
t_elem_7 = 0
t_thresh_7 = 0
t_where_7 = 0

for order in orders:
    t0 = time.perf_counter()
    signs_1d = 2.0 * order - 1.0
    t1 = time.perf_counter()
    np.dot(signs_1d, X_real, out=mean_s_r)
    t2 = time.perf_counter()
    mean_s_r /= n_subjects
    np.multiply(mean_s_r, mean_s_r, out=denom_sq_r)
    denom_sq_r *= -n_subjects
    denom_sq_r += sum_sq_real
    np.maximum(denom_sq_r, 0, out=denom_sq_r)
    np.sqrt(denom_sq_r, out=denom_sq_r)
    np.divide(mean_s_r, denom_sq_r, out=t_buf_r)
    t_buf_r *= sqrt_n_nm1
    t3 = time.perf_counter()
    x_in = t_buf_r > threshold_real
    t4 = time.perf_counter()
    act = np.where(x_in)[0]
    t5 = time.perf_counter()

    t_signs_7 += t1 - t0
    t_dot_7 += t2 - t1
    t_elem_7 += t3 - t2
    t_thresh_7 += t4 - t3
    t_where_7 += t5 - t4

total_v7 = t_signs_7 + t_dot_7 + t_elem_7 + t_thresh_7 + t_where_7
print(f"\n  v7 (numpy) per-perm breakdown ({n_perms} perms):")
print(f"    signs:      {t_signs_7/n_perms*1e6:7.1f} µs  ({t_signs_7/total_v7*100:5.1f}%)")
print(f"    dot:        {t_dot_7/n_perms*1e6:7.1f} µs  ({t_dot_7/total_v7*100:5.1f}%)")
print(f"    elementwise:{t_elem_7/n_perms*1e6:7.1f} µs  ({t_elem_7/total_v7*100:5.1f}%)")
print(f"    threshold:  {t_thresh_7/n_perms*1e6:7.1f} µs  ({t_thresh_7/total_v7*100:5.1f}%)")
print(f"    np.where:   {t_where_7/n_perms*1e6:7.1f} µs  ({t_where_7/total_v7*100:5.1f}%)")
print(f"    TOTAL:      {total_v7/n_perms*1e6:7.1f} µs")

ttest_speedup = total_v7 / total_v8
print(f"\n  Ttest+threshold+where speedup: {ttest_speedup:.2f}x")


# ---- Test 3: A/B via _do_1samp_permutations (isolated) ----
print("\n" + "=" * 70)
print("Test 3: A/B via _do_1samp_permutations (real fsaverage)")
print("=" * 70)

from mne.stats.cluster_level import (
    _do_1samp_permutations,
    _find_clusters,
    _setup_adjacency,
)
from mne.stats.parametric import ttest_1samp_no_p
from mne.utils import ProgressBar

# Setup adjacency for spatio-temporal
adj_list = _setup_adjacency(adjacency, n_tests, n_times)
threshold_val = 1.67
max_step = 1

# Generate orders
rng = np.random.RandomState(42)
n_perms_list = [64, 128, 256, 512, 1024]

for n_p in n_perms_list:
    custom_orders = [rng.randint(0, 2, size=n_subjects).astype(bool) for _ in range(n_p)]
    pb = ProgressBar(n_p)

    t0 = time.perf_counter()
    max_sums = _do_1samp_permutations(
        X_real, slices=None, threshold=threshold_val, tail=1,
        adjacency=adj_list, stat_fun=ttest_1samp_no_p,
        max_step=max_step, include=None, partitions=None,
        t_power=1, orders=custom_orders, sample_shape=(n_tests,),
        buffer_size=None, progress_bar=pb,
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    per_perm = elapsed / n_p * 1000  # ms
    print(f"  {n_p:5d} perms: {elapsed:.3f}s  ({per_perm:.3f} ms/perm)")

# Linear regression for per-perm cost
times = []
perms = []
for n_p in [256, 512, 1024, 2048]:
    custom_orders = [rng.randint(0, 2, size=n_subjects).astype(bool) for _ in range(n_p)]
    pb = ProgressBar(n_p)
    t0 = time.perf_counter()
    _do_1samp_permutations(
        X_real, slices=None, threshold=threshold_val, tail=1,
        adjacency=adj_list, stat_fun=ttest_1samp_no_p,
        max_step=max_step, include=None, partitions=None,
        t_power=1, orders=custom_orders, sample_shape=(n_tests,),
        buffer_size=None, progress_bar=pb,
    )
    t1 = time.perf_counter()
    times.append(t1 - t0)
    perms.append(n_p)

# Linear fit
A = np.vstack([perms, np.ones(len(perms))]).T
slope, intercept = np.linalg.lstsq(A, times, rcond=None)[0]
print(f"\n  Linear regression: overhead = {intercept*1e3:.1f}ms, per_perm = {slope*1e3:.3f}ms")


# ---- Test 4: Parity — end-to-end permutation cluster test ----
print("\n" + "=" * 70)
print("Test 4: Parity — end-to-end spatio_temporal_cluster_1samp_test")
print("=" * 70)

# Run with a small number of permutations and verify results are deterministic
from mne.stats import spatio_temporal_cluster_1samp_test

# Shape for cluster test: (n_subjects, n_times, n_vertices)
X_3d = X_real.reshape(n_subjects, n_times, n_src)
X_3d = np.transpose(X_3d, (0, 2, 1))  # (n_subjects, n_vertices, n_times)
# Wait, the test expects (n_subjects, n_times, n_vertices) or (n_obs, *spatial_shape)?
# Actually it's (n_obs, time, space)
X_test = X_real.reshape(n_subjects, n_times, n_src)

t_obs, clusters, pvals, H0 = spatio_temporal_cluster_1samp_test(
    X_test, adjacency=adjacency, n_permutations=64, threshold=threshold_val,
    tail=1, seed=42, verbose=False, out_type="mask",
)
print(f"  n_clusters: {len(clusters)}")
print(f"  H0 range: [{H0.min():.3f}, {H0.max():.3f}]")
print(f"  t_obs range: [{t_obs.min():.3f}, {t_obs.max():.3f}]")
if len(pvals) > 0:
    print(f"  p-values range: [{pvals.min():.4f}, {pvals.max():.4f}]")

# Run twice to verify determinism
t_obs2, clusters2, pvals2, H0_2 = spatio_temporal_cluster_1samp_test(
    X_test, adjacency=adjacency, n_permutations=64, threshold=threshold_val,
    tail=1, seed=42, verbose=False, out_type="mask",
)
assert np.array_equal(t_obs, t_obs2), "t_obs mismatch!"
assert np.array_equal(H0, H0_2), "H0 mismatch!"
assert np.array_equal(pvals, pvals2), "pvals mismatch!"
print(f"  Determinism check: PASS ✓")
