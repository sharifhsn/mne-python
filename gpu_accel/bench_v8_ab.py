"""A/B comparison: v8 fused ttest vs v7 numpy ttest.

Directly times the inner loop of _do_1samp_permutations to isolate the
ttest change from progress bar and other overhead.
"""

import time
import numpy as np
import sys
sys.path.insert(0, "/Users/sharif/Code/mne-python")

import mne
from mne.stats.cluster_level import (
    _do_1samp_permutations,
    _setup_adjacency,
    _fused_ttest,
    _st_fused_ccl,
)
from mne.stats.parametric import ttest_1samp_no_p
from mne.fixes import has_numba
from mne.utils import ProgressBar

# ---- Setup real fsaverage data ----
data_path = mne.datasets.sample.data_path()
subjects_dir = data_path / "subjects"
src = mne.read_source_spaces(
    subjects_dir / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif",
    verbose=False
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

rng = np.random.RandomState(42)
print(f"Setup: {n_subjects} subjects × {n_src} vertices × {n_times} times = {n_tests:,} tests")
print(f"has_numba = {has_numba}")
print("=" * 70)


# ---- Method A: v8 fused ttest (current code) ----
print("\n--- v8 (fused Numba ttest) ---")
for n_p in [128, 256, 512, 1024, 2048]:
    orders = [rng.randint(0, 2, size=n_subjects).astype(bool) for _ in range(n_p)]

    # Time with internal loop only (no progress bar overhead)
    # We'll call _do_1samp_permutations but with a dummy progress bar
    class DummyPB:
        def update(self, n): pass

    t0 = time.perf_counter()
    result = _do_1samp_permutations(
        X, slices=None, threshold=threshold, tail=1,
        adjacency=adj_list, stat_fun=ttest_1samp_no_p,
        max_step=max_step, include=None, partitions=None,
        t_power=1, orders=orders, sample_shape=(n_tests,),
        buffer_size=None, progress_bar=DummyPB(),
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    per_perm = elapsed / n_p * 1000
    print(f"  {n_p:5d} perms: {elapsed:.3f}s  ({per_perm:.3f} ms/perm)")

# Linear regression
times_v8 = []
perms_v8 = []
for n_p in [256, 512, 1024, 2048]:
    orders = [rng.randint(0, 2, size=n_subjects).astype(bool) for _ in range(n_p)]
    t0 = time.perf_counter()
    _do_1samp_permutations(
        X, slices=None, threshold=threshold, tail=1,
        adjacency=adj_list, stat_fun=ttest_1samp_no_p,
        max_step=max_step, include=None, partitions=None,
        t_power=1, orders=orders, sample_shape=(n_tests,),
        buffer_size=None, progress_bar=DummyPB(),
    )
    t1 = time.perf_counter()
    times_v8.append(t1 - t0)
    perms_v8.append(n_p)

A = np.vstack([perms_v8, np.ones(len(perms_v8))]).T
slope_v8, intercept_v8 = np.linalg.lstsq(A, times_v8, rcond=None)[0]
print(f"  Linear: overhead={intercept_v8*1e3:.1f}ms, per_perm={slope_v8*1e3:.3f}ms")


# ---- Method B: v7 numpy ttest (monkeypatch _fused_ttest to be numpy) ----
print("\n--- v7 (numpy ttest, monkeypatched) ---")

# Save original
import mne.stats.cluster_level as cl
_orig_fused = cl._fused_ttest

# Create a numpy-based replacement that mimics the old behavior
def _numpy_ttest(X_T, signs, sum_sq, inv_n, neg_n, sqrt_n_nm1, t_buf):
    """NumPy version (v7 behavior): dot + 8 elementwise ops."""
    # X_T is (n_vars, n_samp), but np.dot(signs, X) needs X as (n_samp, n_vars)
    # So we need X back. We can get it from X_T.T
    np.dot(signs, X_T.T, out=t_buf)
    t_buf /= len(signs)  # t_buf now = mean_s
    # Need a temp for denom_sq
    denom_sq = t_buf * t_buf  # allocates
    denom_sq *= neg_n
    denom_sq += sum_sq
    np.maximum(denom_sq, 0, out=denom_sq)
    np.sqrt(denom_sq, out=denom_sq)
    np.divide(t_buf, denom_sq, out=t_buf)
    t_buf *= sqrt_n_nm1
    # Handle division by zero
    t_buf[~np.isfinite(t_buf)] = 0.0

cl._fused_ttest = _numpy_ttest

for n_p in [128, 256, 512, 1024, 2048]:
    orders = [rng.randint(0, 2, size=n_subjects).astype(bool) for _ in range(n_p)]
    t0 = time.perf_counter()
    _do_1samp_permutations(
        X, slices=None, threshold=threshold, tail=1,
        adjacency=adj_list, stat_fun=ttest_1samp_no_p,
        max_step=max_step, include=None, partitions=None,
        t_power=1, orders=orders, sample_shape=(n_tests,),
        buffer_size=None, progress_bar=DummyPB(),
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    per_perm = elapsed / n_p * 1000
    print(f"  {n_p:5d} perms: {elapsed:.3f}s  ({per_perm:.3f} ms/perm)")

times_v7 = []
perms_v7 = []
for n_p in [256, 512, 1024, 2048]:
    orders = [rng.randint(0, 2, size=n_subjects).astype(bool) for _ in range(n_p)]
    t0 = time.perf_counter()
    _do_1samp_permutations(
        X, slices=None, threshold=threshold, tail=1,
        adjacency=adj_list, stat_fun=ttest_1samp_no_p,
        max_step=max_step, include=None, partitions=None,
        t_power=1, orders=orders, sample_shape=(n_tests,),
        buffer_size=None, progress_bar=DummyPB(),
    )
    t1 = time.perf_counter()
    times_v7.append(t1 - t0)
    perms_v7.append(n_p)

A = np.vstack([perms_v7, np.ones(len(perms_v7))]).T
slope_v7, intercept_v7 = np.linalg.lstsq(A, times_v7, rcond=None)[0]
print(f"  Linear: overhead={intercept_v7*1e3:.1f}ms, per_perm={slope_v7*1e3:.3f}ms")

# Restore
cl._fused_ttest = _orig_fused

# ---- Summary ----
print("\n" + "=" * 70)
print(f"Per-perm: v8 (fused) = {slope_v8*1e3:.3f}ms, v7 (numpy) = {slope_v7*1e3:.3f}ms")
print(f"Speedup: {slope_v7/slope_v8:.2f}x")
print(f"Savings: {(slope_v7 - slope_v8)*1e3:.1f} µs/perm")
