"""Surgical profiling: account for EVERY microsecond of overhead and per-perm cost.

Instruments every step of spatio_temporal_cluster_1samp_test to find the exact
breakdown of:
  - Fixed overhead: ~136ms (everything except the permutation loop)
  - Per-perm slope: ~0.37ms wall / ~1.40ms single-core

Goal: find ANY remaining optimization opportunity by accounting for 100% of time.

Usage:
    .venv/bin/python gpu_accel/profile_every_us.py
"""

import gc
import os
import sys
import time

import numpy as np
from scipy import sparse

import mne
from mne.fixes import has_numba, jit, prange
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.stats.cluster_level import (
    _batched_fused_ttest,
    _check_fun,
    _cluster_indices_to_mask,
    _cluster_mask_to_indices,
    _find_clusters,
    _find_clusters_1dir,
    _fused_ccl,
    _fused_ttest,
    _get_1samp_orders,
    _get_components,
    _get_partitions_from_adjacency,
    _perm_batch_fast,
    _pval_from_histogram,
    _reshape_clusters,
    _setup_adjacency,
    _st_fused_ccl,
    _threshold_to_indices,
    _threshold_to_indices_neg,
    ttest_1samp_no_p,
)
from mne.parallel import parallel_func
from mne.utils import check_random_state, ProgressBar, split_list, _pl, logger

# Suppress MNE logging
mne.set_log_level("WARNING")

N_PERMS = 2048
N_SUBJECTS = 15
N_TIMES = 15
SEED = 42
THRESHOLD = 1.67
TAIL = 0  # two-tailed (standard usage)
T_POWER = 1
MAX_STEP = 1
BATCH_SIZE = 32
N_WARMUP_RUNS = 2
N_TIMED_RUNS = 5
# Match what the actual function uses:
OUT_TYPE = "indices"
CHECK_DISJOINT = False
BUFFER_SIZE = 1000


def fmt_us(seconds):
    """Format seconds as microseconds with commas."""
    us = seconds * 1e6
    if us >= 1000:
        return f"{us:,.0f} us ({seconds*1e3:.3f} ms)"
    return f"{us:.1f} us"


def fmt_ms(seconds):
    """Format seconds as milliseconds."""
    return f"{seconds*1e3:.3f} ms"


def fmt_pct(part, total):
    """Format as percentage."""
    if total == 0:
        return "N/A"
    return f"{part/total*100:.1f}%"


def setup_data():
    """Set up fsaverage ico-5 data."""
    print("=" * 78)
    print("SETUP")
    print("=" * 78)
    data_path = mne.datasets.sample.data_path()
    subjects_dir = data_path / "subjects"
    src = mne.read_source_spaces(
        subjects_dir / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif",
        verbose=False,
    )
    adjacency = mne.spatial_src_adjacency(src, verbose=False)
    n_src = adjacency.shape[0]

    # Random data with signal (same as benchmarks)
    np.random.seed(123)
    X = np.random.randn(N_SUBJECTS, N_TIMES, n_src) * 0.3
    signal_verts = np.arange(100, 200)
    for t in range(5, 10):
        X[:, t, signal_verts] += 2.0

    n_tests = n_src * N_TIMES
    print(f"  {N_SUBJECTS} subjects x {n_src} vertices x {N_TIMES} times = {n_tests:,} tests")
    print(f"  {N_PERMS} permutations, threshold = {THRESHOLD}, tail = {TAIL}")
    print(f"  has_numba = {has_numba}")
    print(f"  BATCH_SIZE = {BATCH_SIZE}")
    print(f"  CPU count = {os.cpu_count()}")
    print(f"  out_type = {OUT_TYPE}, check_disjoint = {CHECK_DISJOINT}")
    print()
    return X, adjacency, n_src, n_tests


def warmup_jit(X, adjacency):
    """Warmup all Numba JIT functions."""
    print("Warming up JIT...")
    _r = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=64, threshold=THRESHOLD,
        tail=TAIL, seed=SEED, verbose=False, out_type=OUT_TYPE,
        check_disjoint=CHECK_DISJOINT,
    )
    del _r
    gc.collect()
    print("  JIT warmup complete.\n")


def profile_e2e(X, adjacency):
    """End-to-end timing to verify sum of parts."""
    print("=" * 78)
    print("END-TO-END VERIFICATION")
    print("=" * 78)

    times = []
    for run in range(N_TIMED_RUNS):
        gc.collect()
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=N_PERMS,
            threshold=THRESHOLD, tail=TAIL, seed=SEED, verbose=False,
            out_type=OUT_TYPE, check_disjoint=CHECK_DISJOINT,
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if run == 0:
            t_obs, clusters, pvals, H0 = _r
        del _r

    times_arr = np.array(times)
    print(f"  Times: {[f'{t:.3f}' for t in times]}")
    print(f"  Median: {np.median(times_arr):.3f}s")
    print(f"  Min:    {np.min(times_arr):.3f}s")
    print(f"  Max:    {np.max(times_arr):.3f}s")
    print(f"  Per-perm (median): {np.median(times_arr)/N_PERMS*1e3:.3f} ms")
    print(f"  Clusters: {len(clusters)}, sig (p<0.05): {np.sum(pvals < 0.05) if len(pvals) > 0 else 0}")
    return np.median(times_arr)


def profile_overhead_accurately(X_raw, adjacency_raw, n_src, n_tests, n_samp):
    """Profile every overhead component that ACTUALLY runs in the e2e path.

    Follows the exact code path of _permutation_cluster_test() line by line.
    """
    print()
    print("=" * 78)
    print("OVERHEAD PROFILING (only what ACTUALLY runs in e2e path)")
    print("=" * 78)
    timings = {}

    # ==== Step 0: _check_fun (called from permutation_cluster_1samp_test) ====
    t0 = time.perf_counter()
    stat_fun, threshold = _check_fun(X_raw, None, None, TAIL)
    t1 = time.perf_counter()
    timings["0. _check_fun"] = t1 - t0

    # ==== _permutation_cluster_test begins (line 1770) ====

    # Step: _check_option calls (line 1793-1794) -- trivial
    t0 = time.perf_counter()
    from mne.utils import _check_option
    _check_option("out_type", OUT_TYPE, ["mask", "indices"])
    _check_option("tail", TAIL, [-1, 0, 1])
    t1 = time.perf_counter()
    timings["1. _check_option calls"] = t1 - t0

    # Step: X reshape (line 1810-1821)
    X_list = [X_raw]
    t0 = time.perf_counter()
    X_list = [x[:, np.newaxis] if x.ndim == 1 else x for x in X_list]
    n_samples = X_list[0].shape[0]
    n_times = X_list[0].shape[1]
    sample_shape = X_list[0].shape[1:]
    X_list = [np.reshape(x, (x.shape[0], -1)) for x in X_list]
    n_tests_local = X_list[0].shape[1]
    t1 = time.perf_counter()
    timings["2. X reshape/flatten"] = t1 - t0

    # Step: _setup_adjacency (line 1824)
    t0 = time.perf_counter()
    adjacency = _setup_adjacency(adjacency_raw.copy(), n_tests_local, n_times)
    t1 = time.perf_counter()
    timings["3. _setup_adjacency TOTAL"] = t1 - t0

    # Break down _setup_adjacency:
    adj_cpy = adjacency_raw.copy()
    t0 = time.perf_counter()
    adj_csr = (adj_cpy + adj_cpy.transpose()).tocsr()
    t1 = time.perf_counter()
    timings["   3a. sparse add + tocsr"] = t1 - t0

    t0 = time.perf_counter()
    adj_lists = [
        adj_csr.indices[adj_csr.indptr[i] : adj_csr.indptr[i + 1]]
        for i in range(len(adj_csr.indptr) - 1)
    ]
    t1 = time.perf_counter()
    timings["   3b. CSR -> neighbor lists"] = t1 - t0
    del adj_csr, adj_lists, adj_cpy

    # Step: stat_fun (initial t_obs) (line 1831)
    t0 = time.perf_counter()
    t_obs = stat_fun(*X_list)
    t1 = time.perf_counter()
    timings["4. ttest_1samp_no_p (initial t_obs)"] = t1 - t0

    # Step: buffer_size verification -- SKIPPED for built-in stat fns (line 1838)
    t0 = time.perf_counter()
    _is_builtin = (stat_fun is ttest_1samp_no_p)
    t1 = time.perf_counter()
    timings["5. buffer_size check (identity, skipped)"] = t1 - t0

    # Step: include = None (line 1863-1866)
    include = None

    # Step: check_disjoint = False => partitions = None (line 1869-1872)
    # NOT called when check_disjoint=False
    timings["6. get_partitions (SKIPPED, check_disjoint=False)"] = 0.0

    # Step: _find_clusters (initial clustering) (line 1874)
    t0 = time.perf_counter()
    clusters, cluster_stats = _find_clusters(
        t_obs, threshold, TAIL, adjacency,
        max_step=MAX_STEP, include=include, partitions=None,
        t_power=T_POWER, show_info=True,
    )
    t1 = time.perf_counter()
    timings["7. _find_clusters (initial) TOTAL"] = t1 - t0
    n_initial_clusters = len(clusters)

    # Break down _find_clusters for adjacency=list path with Numba:
    if TAIL == 0:
        x_ins = [t_obs > threshold, t_obs < -threshold]
    elif TAIL == 1:
        x_ins = [t_obs > threshold]
    else:
        x_ins = [t_obs < threshold]

    t0 = time.perf_counter()
    _ = [t_obs > threshold, t_obs < -threshold]
    t1 = time.perf_counter()
    timings["   7a. threshold comparisons"] = t1 - t0

    for tail_idx, x_in in enumerate(x_ins):
        if not np.any(x_in):
            continue
        tail_name = "pos" if tail_idx == 0 else "neg"
        _n_src_local = len(adjacency)

        # All the CSR setup done inside _find_clusters_1dir for list adj
        t0 = time.perf_counter()
        _lengths = np.array([len(a) for a in adjacency])
        _indptr = np.zeros(_n_src_local + 1, dtype=np.intp)
        np.cumsum(_lengths, out=_indptr[1:])
        _indices = np.concatenate(adjacency).astype(np.intp)
        _flat_map = -np.ones(n_tests_local, dtype=np.intp)
        t1 = time.perf_counter()
        timings[f"   7b. [{tail_name}] CSR setup"] = t1 - t0

        t0 = time.perf_counter()
        active_idx = np.where(x_in)[0].astype(np.intp)
        t1 = time.perf_counter()
        n_active = len(active_idx)
        timings[f"   7c. [{tail_name}] np.where (n={n_active})"] = t1 - t0

        t0 = time.perf_counter()
        components = _st_fused_ccl(
            active_idx, n_active, _flat_map,
            _indptr, _indices, _n_src_local, MAX_STEP,
        )
        t1 = time.perf_counter()
        timings[f"   7d. [{tail_name}] _st_fused_ccl"] = t1 - t0

        t0 = time.perf_counter()
        order = np.argsort(components, kind="stable")
        counts = np.bincount(components)
        splits = np.cumsum(counts[:-1])
        global_order = active_idx[order]
        cluster_list = list(np.split(global_order, splits))
        t1 = time.perf_counter()
        timings[f"   7e. [{tail_name}] argsort/split clusters"] = t1 - t0

        t0 = time.perf_counter()
        all_idx = np.concatenate(cluster_list)
        lengths_ = np.array([len(c) for c in cluster_list])
        offsets = np.empty(len(cluster_list), dtype=np.intp)
        offsets[0] = 0
        np.cumsum(lengths_[:-1], out=offsets[1:])
        sums = np.add.reduceat(t_obs[all_idx], offsets)
        t1 = time.perf_counter()
        timings[f"   7f. [{tail_name}] reduceat sums"] = t1 - t0

    n_supra_pos = np.sum(x_ins[0])
    n_supra_neg = np.sum(x_ins[1]) if len(x_ins) > 1 else 0
    print(f"  Supra-threshold: pos={n_supra_pos}, neg={n_supra_neg}, "
          f"total={n_supra_pos+n_supra_neg} ({(n_supra_pos+n_supra_neg)/n_tests_local*100:.1f}%)")

    # Step: t_obs reshape (line 1888) -- view only
    t0 = time.perf_counter()
    t_obs = t_obs.reshape(sample_shape)
    t1 = time.perf_counter()
    timings["8. t_obs reshape (view)"] = t1 - t0

    # Step: cluster format conversion (line 1899-1908)
    # With adjacency=list and out_type="indices":
    # adjacency is not None and not False => enters the block
    # out_type == "indices" (not "mask") => SKIPS _cluster_indices_to_mask
    timings["9. cluster_format_conv (SKIPPED for indices)"] = 0.0

    # Step: _get_1samp_orders (line 1920)
    rng = check_random_state(SEED)
    t0 = time.perf_counter()
    orders, n_perms_actual, extra = _get_1samp_orders(n_samples, N_PERMS, TAIL, rng)
    t1 = time.perf_counter()
    timings["10. _get_1samp_orders"] = t1 - t0

    # Step: parallel_func (line 1932)
    from mne.stats.cluster_level import _do_1samp_permutations
    t0 = time.perf_counter()
    parallel, my_do_perm_func, n_jobs = parallel_func(
        _do_1samp_permutations, 1, verbose=False
    )
    t1 = time.perf_counter()
    timings["11. parallel_func setup"] = t1 - t0

    # Step: Check if clusters exist (line 1936)
    # clusters is non-empty => continue

    # ==== Inside _do_1samp_permutations (called via parallel) ====
    # The overhead here is the setup BEFORE the batch loop
    X_2d = X_list[0]

    # buffer_size check (line 1384)
    t0 = time.perf_counter()
    buffer_size_local = BUFFER_SIZE
    if buffer_size_local is not None and n_tests_local <= buffer_size_local:
        buffer_size_local = None
    t1 = time.perf_counter()
    timings["12. buffer_size clamp"] = t1 - t0

    # max_cluster_sums alloc (line 1388)
    t0 = time.perf_counter()
    max_cluster_sums = np.empty(len(orders), dtype=np.double)
    t1 = time.perf_counter()
    timings["13. alloc max_cluster_sums"] = t1 - t0

    # _use_fast_ttest precomputation (lines 1395-1438)
    t0 = time.perf_counter()
    _sum_sq = np.sum(X_2d**2, axis=0)
    t1 = time.perf_counter()
    timings["14. precompute sum_sq"] = t1 - t0

    t0 = time.perf_counter()
    _sqrt_n_nm1 = np.sqrt(n_samples * (n_samples - 1))
    _inv_n = 1.0 / n_samples
    _neg_n = -float(n_samples)
    t1 = time.perf_counter()
    timings["15. scalar constants"] = t1 - t0

    t0 = time.perf_counter()
    _X_T = np.ascontiguousarray(X_2d.T)
    t1 = time.perf_counter()
    timings["16. X.T contiguous copy"] = t1 - t0
    print(f"  X_T: {_X_T.shape}, {_X_T.nbytes/1e6:.1f} MB")

    t0 = time.perf_counter()
    _t_buf = np.empty(n_tests_local, dtype=np.float64)
    _idx_buf = np.empty(n_tests_local, dtype=np.intp)
    t1 = time.perf_counter()
    timings["17. alloc t_buf + idx_buf"] = t1 - t0

    # JIT warmup calls (lines 1408-1420)
    # These are already warm so nearly free, but measure anyway
    _warmup_X = _X_T[:2]
    _warmup_sq = _sum_sq[:2]
    _warmup_t = np.empty(2, dtype=np.float64)
    _warmup_s = np.ones(n_samples, dtype=np.float64)
    t0 = time.perf_counter()
    _fused_ttest(_warmup_X, _warmup_s, _warmup_sq, _inv_n, _neg_n, _sqrt_n_nm1, _warmup_t)
    _threshold_to_indices(_warmup_t, 0.0, np.empty(2, dtype=np.intp))
    _threshold_to_indices_neg(_warmup_t, 0.0, np.empty(2, dtype=np.intp))
    t1 = time.perf_counter()
    timings["18. JIT warmup calls (already warm)"] = t1 - t0

    # Batched ttest setup (lines 1425-1438)
    _BATCH = BATCH_SIZE
    t0 = time.perf_counter()
    _all_signs = 2.0 * np.array(orders, dtype=np.float64) - 1.0
    t1 = time.perf_counter()
    timings["19. sign vector pre-computation"] = t1 - t0

    t0 = time.perf_counter()
    _t_batch = np.empty((_BATCH, n_tests_local), dtype=np.float64)
    t1 = time.perf_counter()
    timings["20. alloc t_batch"] = t1 - t0

    # Batched ttest warmup (lines 1431-1438) -- already warm
    _warmup_sb = np.ones((2, n_samples), dtype=np.float64)
    _warmup_tb = np.empty((2, 2), dtype=np.float64)
    t0 = time.perf_counter()
    _batched_fused_ttest(_warmup_X, _warmup_sb, _warmup_sq,
                         _inv_n, _neg_n, _sqrt_n_nm1, _warmup_tb, 2)
    t1 = time.perf_counter()
    timings["21. batched ttest warmup (already warm)"] = t1 - t0
    del _warmup_X, _warmup_sq, _warmup_t, _warmup_s, _warmup_sb, _warmup_tb

    # CSR conversion for _st_fused_ccl (lines 1450-1466)
    t0 = time.perf_counter()
    _st_n_src = len(adjacency)
    _st_n_times = n_tests_local // _st_n_src
    _st_lengths = np.array([len(a) for a in adjacency])
    _st_indptr = np.zeros(_st_n_src + 1, dtype=np.intp)
    np.cumsum(_st_lengths, out=_st_indptr[1:])
    _st_indices = np.concatenate(adjacency).astype(np.intp)
    _st_flat_map = -np.ones(n_tests_local, dtype=np.intp)
    t1 = time.perf_counter()
    timings["22. CSR conversion (perm loop)"] = t1 - t0

    # _st_fused_ccl warmup (lines 1461-1466) -- already warm
    _dummy_idx = np.array([0], dtype=np.intp)
    t0 = time.perf_counter()
    _st_fused_ccl(_dummy_idx, 1, _st_flat_map, _st_indptr, _st_indices, _st_n_src, MAX_STEP)
    t1 = time.perf_counter()
    timings["23. _st_fused_ccl warmup (already warm)"] = t1 - t0
    del _dummy_idx

    # Parallel work buffers (lines 1478-1480)
    t0 = time.perf_counter()
    _par_idx_bufs = np.empty((_BATCH, n_tests_local), dtype=np.intp)
    _par_flat_maps = -np.ones((_BATCH, n_tests_local), dtype=np.intp)
    _par_sums_bufs = np.empty((_BATCH, n_tests_local), dtype=np.float64)
    t1 = time.perf_counter()
    timings["24. alloc parallel work bufs"] = t1 - t0
    total_buf_mb = (_par_idx_bufs.nbytes + _par_flat_maps.nbytes + _par_sums_bufs.nbytes) / 1e6
    print(f"  Parallel work bufs: {total_buf_mb:.1f} MB")

    # _perm_batch_fast warmup (lines 1482-1494) -- already warm
    _warmup_tb2 = np.zeros((2, 2), dtype=np.float64)
    _warmup_ms = np.empty(2, dtype=np.float64)
    _warmup_idx = np.empty((2, 2), dtype=np.intp)
    _warmup_fm = -np.ones((2, n_tests_local), dtype=np.intp)
    _warmup_sb = np.empty((2, n_tests_local), dtype=np.float64)
    t0 = time.perf_counter()
    _perm_batch_fast(
        _warmup_tb2, 2, 0, 0.0, 1, 1.0,
        _warmup_idx, _warmup_fm,
        _st_indptr, _st_indices, _st_n_src, MAX_STEP,
        _warmup_ms, _warmup_sb,
    )
    t1 = time.perf_counter()
    timings["25. _perm_batch_fast warmup (already warm)"] = t1 - t0
    del _warmup_tb2, _warmup_ms, _warmup_idx, _warmup_fm, _warmup_sb

    # ProgressBar creation (line 1958)
    t0 = time.perf_counter()
    pb = ProgressBar(iterable=range(len(orders)), mesg=f"Permuting{extra}")
    t1 = time.perf_counter()
    timings["26. ProgressBar creation"] = t1 - t0
    del pb

    # ==== POST-PERMUTATION OVERHEAD ====

    # H0.insert + concatenate (lines 1981-1988)
    H0_fake = [max_cluster_sums]
    t0 = time.perf_counter()
    if TAIL == -1:
        orig = cluster_stats.min()
    elif TAIL == 1:
        orig = cluster_stats.max()
    else:
        orig = abs(cluster_stats).max()
    H0_copy = list(H0_fake)
    H0_copy.insert(0, [orig])
    H0_concat = np.concatenate(H0_copy)
    t1 = time.perf_counter()
    timings["27. H0 concat + insert"] = t1 - t0

    # _pval_from_histogram (line 1990)
    t0 = time.perf_counter()
    cluster_pv = _pval_from_histogram(cluster_stats, H0_concat, TAIL)
    t1 = time.perf_counter()
    timings["28. _pval_from_histogram"] = t1 - t0

    # step_down_p loop overhead (lines 1993-2001) -- single iteration, no removal
    t0 = time.perf_counter()
    to_remove = np.where(cluster_pv < 0)[0]  # step_down_p=0 => nothing removed
    n_removed = to_remove.size
    step_down_include = np.ones(n_tests_local, dtype=bool)
    t1 = time.perf_counter()
    timings["29. step_down check"] = t1 - t0

    # _reshape_clusters (line 2014)
    # With out_type="indices" and adjacency=list: clusters are index arrays
    # _reshape_clusters calls np.unravel_index for each cluster
    clusters_copy = [c.copy() for c in clusters]
    t0 = time.perf_counter()
    _ = _reshape_clusters(clusters_copy, sample_shape)
    t1 = time.perf_counter()
    timings["30. _reshape_clusters"] = t1 - t0
    print(f"  n_clusters to reshape: {len(clusters_copy)}")

    # ==== PRINT OVERHEAD SUMMARY ====
    print()
    print("-" * 78)
    print("OVERHEAD BREAKDOWN (only items that actually execute):")
    print("-" * 78)
    total_overhead = 0
    for name, t in timings.items():
        indent = "  " if not name.startswith(" ") else ""
        is_sub = name.startswith("   ")
        if not is_sub:
            total_overhead += t
        marker = "" if t < 0.001 else " ***" if t > 0.005 else " **"
        print(f"{indent}{name:55s} {fmt_us(t):>25s}{marker}")

    print(f"\n{'TOTAL measured overhead':55s} {fmt_us(total_overhead):>25s}")
    print(f"  Initial clusters found: {n_initial_clusters}")

    return (adjacency, t_obs, clusters, cluster_stats, orders,
            _X_T, _sum_sq, _inv_n, _neg_n, _sqrt_n_nm1,
            _all_signs, _t_batch, _idx_buf,
            _st_indptr, _st_indices, _st_n_src,
            _par_idx_bufs, _par_flat_maps, _par_sums_bufs,
            max_cluster_sums, timings, n_tests_local, sample_shape)


def profile_permloop(X_flat, adjacency, n_tests, n_samp,
                     _X_T, _sum_sq, _inv_n, _neg_n, _sqrt_n_nm1,
                     _all_signs, _t_batch, _idx_buf,
                     _st_indptr, _st_indices, _st_n_src,
                     _par_idx_bufs, _par_flat_maps, _par_sums_bufs,
                     max_cluster_sums, orders):
    """Profile the permutation loop with per-batch timing."""
    print()
    print("=" * 78)
    print("PERMUTATION LOOP PROFILING")
    print("=" * 78)

    n_perms = len(orders)
    n_batches = (n_perms + BATCH_SIZE - 1) // BATCH_SIZE
    threshold = THRESHOLD

    # Timing accumulators
    batch_ttest_times = []
    batch_inner_times = []
    batch_python_overhead = []
    batch_total_times = []
    batch_sizes = []
    batch_slice_times = []
    batch_progress_times = []

    t_loop_start = time.perf_counter()

    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, n_perms)
        n_batch = batch_end - batch_start
        batch_sizes.append(n_batch)

        t_batch_wall_start = time.perf_counter()

        # Array slice for signs
        t_s0 = time.perf_counter()
        signs_slice = _all_signs[batch_start:batch_end]
        t_s1 = time.perf_counter()
        batch_slice_times.append(t_s1 - t_s0)

        # 1. Batched fused ttest
        t0 = time.perf_counter()
        _batched_fused_ttest(
            _X_T, signs_slice, _sum_sq,
            _inv_n, _neg_n, _sqrt_n_nm1, _t_batch, n_batch,
        )
        t1 = time.perf_counter()
        batch_ttest_times.append(t1 - t0)

        # 2. Fused inner loop (threshold + CCL + bincount + argmax)
        t2 = time.perf_counter()
        _perm_batch_fast(
            _t_batch, n_batch, batch_start,
            threshold, TAIL, T_POWER,
            _par_idx_bufs, _par_flat_maps,
            _st_indptr, _st_indices, _st_n_src, MAX_STEP,
            max_cluster_sums, _par_sums_bufs,
        )
        t3 = time.perf_counter()
        batch_inner_times.append(t3 - t2)

        # ProgressBar update (simulated)
        t_p0 = time.perf_counter()
        # In real code: progress_bar.update(batch_end)
        t_p1 = time.perf_counter()
        batch_progress_times.append(t_p1 - t_p0)

        t_batch_wall_end = time.perf_counter()
        batch_total_times.append(t_batch_wall_end - t_batch_wall_start)

        # Python overhead = total - (ttest + inner)
        python_oh = (t_batch_wall_end - t_batch_wall_start) - (t1 - t0) - (t3 - t2)
        batch_python_overhead.append(python_oh)

    t_loop_end = time.perf_counter()
    total_loop_time = t_loop_end - t_loop_start

    # Analysis
    ttest_total = sum(batch_ttest_times)
    inner_total = sum(batch_inner_times)
    python_total = sum(batch_python_overhead)
    slice_total = sum(batch_slice_times)
    accounted = ttest_total + inner_total + python_total
    gap = total_loop_time - accounted

    print(f"\n  Total loop wall time:     {fmt_ms(total_loop_time)}")
    print(f"  Total batches:            {n_batches}")
    print(f"  Total perms:              {n_perms}")
    print()
    print(f"  _batched_fused_ttest:     {fmt_ms(ttest_total):>18s}  "
          f"({fmt_pct(ttest_total, total_loop_time)})")
    print(f"  _perm_batch_fast:         {fmt_ms(inner_total):>18s}  "
          f"({fmt_pct(inner_total, total_loop_time)})")
    print(f"  Python loop overhead:     {fmt_ms(python_total):>18s}  "
          f"({fmt_pct(python_total, total_loop_time)})")
    print(f"    (of which array slice): {fmt_ms(slice_total):>18s}")
    print(f"  Measurement gap:          {fmt_ms(gap):>18s}  "
          f"({fmt_pct(gap, total_loop_time)})")
    print()

    # Per-perm breakdown
    per_perm_ttest = ttest_total / n_perms
    per_perm_inner = inner_total / n_perms
    per_perm_python = python_total / n_perms
    per_perm_total = total_loop_time / n_perms

    print("  Per-perm breakdown:")
    print(f"    _batched_fused_ttest:   {per_perm_ttest*1e6:>10.1f} us")
    print(f"    _perm_batch_fast:       {per_perm_inner*1e6:>10.1f} us")
    print(f"    Python overhead:        {per_perm_python*1e6:>10.1f} us")
    print(f"    TOTAL per-perm:         {per_perm_total*1e6:>10.1f} us")
    print()

    # Per-batch statistics
    ttest_arr = np.array(batch_ttest_times)
    inner_arr = np.array(batch_inner_times)
    python_arr = np.array(batch_python_overhead)
    total_arr = np.array(batch_total_times)

    print("  Per-batch statistics (ms):")
    print(f"    {'':20s} {'median':>10s} {'mean':>10s} {'min':>10s} {'max':>10s} {'std':>10s}")
    for name, arr in [("ttest", ttest_arr), ("inner", inner_arr),
                       ("python_oh", python_arr), ("total", total_arr)]:
        print(f"    {name:20s} {np.median(arr)*1e3:>10.3f} {np.mean(arr)*1e3:>10.3f} "
              f"{np.min(arr)*1e3:>10.3f} {np.max(arr)*1e3:>10.3f} {np.std(arr)*1e3:>10.3f}")

    return {
        "total_loop": total_loop_time,
        "ttest_total": ttest_total,
        "inner_total": inner_total,
        "python_total": python_total,
        "gap": gap,
        "per_perm_total": per_perm_total,
        "per_perm_ttest": per_perm_ttest,
        "per_perm_inner": per_perm_inner,
        "per_perm_python": per_perm_python,
        "batch_ttest": ttest_arr,
        "batch_inner": inner_arr,
    }


def profile_ttest_micro(n_tests, n_samp,
                        _X_T, _sum_sq, _inv_n, _neg_n, _sqrt_n_nm1,
                        _all_signs):
    """Micro-profile the batched ttest to understand its cost."""
    print()
    print("=" * 78)
    print("MICRO-PROFILE: _batched_fused_ttest")
    print("=" * 78)

    n_vars = n_tests
    _t_batch = np.empty((BATCH_SIZE, n_vars), dtype=np.float64)

    # Time single-perm _fused_ttest for comparison
    _t_buf = np.empty(n_vars, dtype=np.float64)
    signs = _all_signs[0]
    times_single = []
    for _ in range(50):
        t0 = time.perf_counter()
        _fused_ttest(_X_T, signs, _sum_sq, _inv_n, _neg_n, _sqrt_n_nm1, _t_buf)
        t1 = time.perf_counter()
        times_single.append(t1 - t0)

    # Time batched
    times_batched = []
    for batch_start in range(0, min(320, len(_all_signs)), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(_all_signs))
        n_batch = batch_end - batch_start
        t0 = time.perf_counter()
        _batched_fused_ttest(
            _X_T, _all_signs[batch_start:batch_end], _sum_sq,
            _inv_n, _neg_n, _sqrt_n_nm1, _t_batch, n_batch,
        )
        t1 = time.perf_counter()
        times_batched.append((t1 - t0, n_batch))

    single_median = np.median(times_single)
    batch_per_perm = np.median([t/n for t, n in times_batched])

    print(f"  Single _fused_ttest:      {single_median*1e6:.1f} us/perm")
    print(f"  Batched (B={BATCH_SIZE}):             {batch_per_perm*1e6:.1f} us/perm")
    print(f"  Batching speedup:         {single_median/batch_per_perm:.2f}x")
    print(f"  X_T memory:               {_X_T.nbytes / 1e6:.1f} MB")

    # Memory bandwidth estimate
    bytes_per_batch = _X_T.nbytes  # read X_T once
    bytes_per_batch += BATCH_SIZE * n_samp * 8  # read signs
    bytes_per_batch += BATCH_SIZE * n_vars * 8  # write t_batch
    bytes_per_batch += n_vars * 8  # read sum_sq
    median_batch_time = np.median([t for t, _ in times_batched])
    bw_gb = (bytes_per_batch / 1e9) / median_batch_time
    print(f"  Effective bandwidth:      {bw_gb:.1f} GB/s")


def profile_ccl_micro(n_tests, _X_T, _sum_sq, _inv_n, _neg_n, _sqrt_n_nm1,
                      _all_signs, _st_indptr, _st_indices, _st_n_src):
    """Micro-profile CCL to understand its cost."""
    print()
    print("=" * 78)
    print("MICRO-PROFILE: _st_fused_ccl (CCL)")
    print("=" * 78)

    n_vars = n_tests
    _t_buf = np.empty(n_vars, dtype=np.float64)
    signs = _all_signs[0]
    _fused_ttest(_X_T, signs, _sum_sq, _inv_n, _neg_n, _sqrt_n_nm1, _t_buf)

    threshold = THRESHOLD
    _idx_buf = np.empty(n_vars, dtype=np.intp)
    n_active_pos = _threshold_to_indices(_t_buf, threshold, _idx_buf)
    print(f"  n_active (pos): {n_active_pos} / {n_vars} ({n_active_pos/n_vars*100:.1f}%)")

    n_active_neg = _threshold_to_indices_neg(_t_buf, -threshold, _idx_buf)
    print(f"  n_active (neg): {n_active_neg} / {n_vars} ({n_active_neg/n_vars*100:.1f}%)")

    # Time individual components
    flat_map = -np.ones(n_vars, dtype=np.intp)

    # Threshold scan
    times_thresh_pos = []
    times_thresh_neg = []
    for _ in range(200):
        t0 = time.perf_counter()
        n = _threshold_to_indices(_t_buf, threshold, _idx_buf)
        t1 = time.perf_counter()
        times_thresh_pos.append(t1 - t0)

        t0 = time.perf_counter()
        n = _threshold_to_indices_neg(_t_buf, -threshold, _idx_buf)
        t1 = time.perf_counter()
        times_thresh_neg.append(t1 - t0)

    # CCL
    n_act = _threshold_to_indices(_t_buf, threshold, _idx_buf)
    times_ccl = []
    for _ in range(200):
        t0 = time.perf_counter()
        comps = _st_fused_ccl(
            _idx_buf[:n_act], n_act, flat_map,
            _st_indptr, _st_indices, _st_n_src, MAX_STEP,
        )
        t1 = time.perf_counter()
        times_ccl.append(t1 - t0)

    # bincount + argmax
    times_bc = []
    for _ in range(200):
        t0 = time.perf_counter()
        sums = np.bincount(comps, weights=_t_buf[_idx_buf[:n_act]])
        idx_max = np.argmax(np.abs(sums))
        best = sums[idx_max]
        t1 = time.perf_counter()
        times_bc.append(t1 - t0)

    print(f"\n  Component timings (median of 200 runs):")
    print(f"    threshold scan (pos):   {np.median(times_thresh_pos)*1e6:.1f} us")
    print(f"    threshold scan (neg):   {np.median(times_thresh_neg)*1e6:.1f} us")
    print(f"    _st_fused_ccl:          {np.median(times_ccl)*1e6:.1f} us (n_active={n_act})")
    print(f"    bincount + argmax:      {np.median(times_bc)*1e6:.1f} us")
    n_comps = len(np.unique(comps))
    print(f"    n_components:           {n_comps}")

    # CCL scaling
    print("\n  CCL scaling with density:")
    print(f"    {'perm':>5s} {'density':>10s} {'n_active':>10s} {'ccl_us':>10s} {'n_comps':>10s}")
    for perm_idx in range(0, min(10, len(_all_signs))):
        signs = _all_signs[perm_idx]
        _fused_ttest(_X_T, signs, _sum_sq, _inv_n, _neg_n, _sqrt_n_nm1, _t_buf)
        n_act = _threshold_to_indices(_t_buf, threshold, _idx_buf)
        if n_act == 0:
            continue
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            comps = _st_fused_ccl(
                _idx_buf[:n_act], n_act, flat_map,
                _st_indptr, _st_indices, _st_n_src, MAX_STEP,
            )
            t1 = time.perf_counter()
            times.append(t1 - t0)
        n_c = len(np.unique(comps))
        print(f"    {perm_idx:>5d} {n_act/n_vars*100:>9.1f}% {n_act:>10d} "
              f"{np.median(times)*1e6:>10.1f} {n_c:>10d}")


def profile_perm_batch_fast_micro(n_tests, _X_T, _sum_sq, _inv_n, _neg_n,
                                   _sqrt_n_nm1, _all_signs, _t_batch,
                                   _st_indptr, _st_indices, _st_n_src,
                                   _par_idx_bufs, _par_flat_maps,
                                   _par_sums_bufs, max_cluster_sums):
    """Micro-profile _perm_batch_fast alone."""
    print()
    print("=" * 78)
    print("MICRO-PROFILE: _perm_batch_fast (fused inner loop)")
    print("=" * 78)

    threshold = THRESHOLD
    n_vars = n_tests

    # Compute ttest for first batch
    _batched_fused_ttest(
        _X_T, _all_signs[:BATCH_SIZE], _sum_sq,
        _inv_n, _neg_n, _sqrt_n_nm1, _t_batch, BATCH_SIZE,
    )

    # Time _perm_batch_fast alone
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        _perm_batch_fast(
            _t_batch, BATCH_SIZE, 0,
            threshold, TAIL, T_POWER,
            _par_idx_bufs, _par_flat_maps,
            _st_indptr, _st_indices, _st_n_src, MAX_STEP,
            max_cluster_sums, _par_sums_bufs,
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times_arr = np.array(times)
    median_batch = np.median(times_arr)
    per_perm = median_batch / BATCH_SIZE

    print(f"  Batch of {BATCH_SIZE}: median {median_batch*1e3:.3f} ms, "
          f"per-perm {per_perm*1e6:.1f} us")
    print(f"  Min: {np.min(times_arr)*1e3:.3f} ms, Max: {np.max(times_arr)*1e3:.3f} ms")

    # Check supra-threshold density for each perm in batch
    pos_counts = []
    neg_counts = []
    for b in range(BATCH_SIZE):
        n_pos = np.sum(_t_batch[b] > threshold)
        n_neg = np.sum(_t_batch[b] < -threshold)
        pos_counts.append(n_pos)
        neg_counts.append(n_neg)

    pos_arr = np.array(pos_counts)
    neg_arr = np.array(neg_counts)
    total_active = np.mean(pos_arr) + np.mean(neg_arr)
    print(f"  Density (pos): mean={np.mean(pos_arr):.0f}, range=[{np.min(pos_arr)}, {np.max(pos_arr)}]")
    print(f"  Density (neg): mean={np.mean(neg_arr):.0f}, range=[{np.min(neg_arr)}, {np.max(neg_arr)}]")
    print(f"  Total active per perm: {total_active:.0f} ({total_active/n_tests*100:.1f}%)")
    print(f"  With tail=0, each perm processes BOTH tails => 2x CCL calls")


def main():
    # ================================================================
    # SETUP
    # ================================================================
    X, adjacency_raw, n_src, n_tests = setup_data()
    n_samp = X.shape[0]
    sample_shape = X.shape[1:]  # (N_TIMES, n_src)

    # ================================================================
    # JIT WARMUP
    # ================================================================
    warmup_jit(X, adjacency_raw)

    # ================================================================
    # END-TO-END BASELINE
    # ================================================================
    e2e_median = profile_e2e(X, adjacency_raw)

    # ================================================================
    # OVERHEAD PROFILING (accurate, only what runs in e2e path)
    # ================================================================
    (adjacency, t_obs, clusters, cluster_stats, orders,
     _X_T, _sum_sq, _inv_n, _neg_n, _sqrt_n_nm1,
     _all_signs, _t_batch, _idx_buf,
     _st_indptr, _st_indices, _st_n_src,
     _par_idx_bufs, _par_flat_maps, _par_sums_bufs,
     max_cluster_sums, overhead_timings, n_tests_actual,
     sample_shape_actual) = profile_overhead_accurately(
        X, adjacency_raw, n_src, n_tests, n_samp
    )

    # ================================================================
    # PERMUTATION LOOP PROFILING
    # ================================================================
    perm_results = profile_permloop(
        X, adjacency, n_tests, n_samp,
        _X_T, _sum_sq, _inv_n, _neg_n, _sqrt_n_nm1,
        _all_signs, _t_batch, _idx_buf,
        _st_indptr, _st_indices, _st_n_src,
        _par_idx_bufs, _par_flat_maps, _par_sums_bufs,
        max_cluster_sums, orders,
    )

    # ================================================================
    # MICRO-PROFILES
    # ================================================================
    profile_ttest_micro(
        n_tests, n_samp, _X_T, _sum_sq, _inv_n, _neg_n, _sqrt_n_nm1,
        _all_signs,
    )

    profile_ccl_micro(
        n_tests, _X_T, _sum_sq, _inv_n, _neg_n, _sqrt_n_nm1,
        _all_signs, _st_indptr, _st_indices, _st_n_src,
    )

    profile_perm_batch_fast_micro(
        n_tests, _X_T, _sum_sq, _inv_n, _neg_n, _sqrt_n_nm1,
        _all_signs, _t_batch, _st_indptr, _st_indices, _st_n_src,
        _par_idx_bufs, _par_flat_maps, _par_sums_bufs, max_cluster_sums,
    )

    # ================================================================
    # FINAL SUMMARY: ACCOUNT FOR 100%
    # ================================================================
    print()
    print("=" * 78)
    print("FINAL ACCOUNTING: WHERE DOES EVERY MICROSECOND GO?")
    print("=" * 78)

    # Sum up only top-level overhead items (not sub-items)
    overhead_sum = sum(
        t for name, t in overhead_timings.items()
        if not name.startswith("   ")
    )
    loop_total = perm_results["total_loop"]
    total_accounted = overhead_sum + loop_total
    gap_from_e2e = e2e_median - total_accounted

    print(f"\n  End-to-end median:        {fmt_ms(e2e_median)}")
    print(f"  Overhead (measured):      {fmt_ms(overhead_sum)}")
    print(f"  Perm loop (measured):     {fmt_ms(loop_total)}")
    print(f"  Sum (overhead + loop):    {fmt_ms(total_accounted)}")
    print(f"  GAP (e2e - sum):          {fmt_ms(gap_from_e2e)}")
    print(f"  GAP as % of e2e:          {fmt_pct(gap_from_e2e, e2e_median)}")

    print(f"\n  === OVERHEAD BREAKDOWN ({fmt_ms(overhead_sum)}) ===")
    sorted_oh = sorted(
        [(name, t) for name, t in overhead_timings.items() if not name.startswith("   ")],
        key=lambda x: x[1],
        reverse=True,
    )
    for name, t in sorted_oh:
        if t > 0.0001:  # > 0.1ms
            print(f"    {name:50s} {fmt_ms(t):>15s}  ({fmt_pct(t, overhead_sum)})")

    print(f"\n  === PER-PERM BREAKDOWN ({perm_results['per_perm_total']*1e6:.1f} us/perm) ===")
    print(f"    _batched_fused_ttest:   {perm_results['per_perm_ttest']*1e6:>10.1f} us  "
          f"({fmt_pct(perm_results['per_perm_ttest'], perm_results['per_perm_total'])})")
    print(f"    _perm_batch_fast:       {perm_results['per_perm_inner']*1e6:>10.1f} us  "
          f"({fmt_pct(perm_results['per_perm_inner'], perm_results['per_perm_total'])})")
    print(f"    Python overhead:        {perm_results['per_perm_python']*1e6:>10.1f} us  "
          f"({fmt_pct(perm_results['per_perm_python'], perm_results['per_perm_total'])})")

    # Identify optimization opportunities
    print(f"\n  === OPTIMIZATION OPPORTUNITIES ===")
    opportunities = []
    for name, t in sorted_oh:
        if t > 0.001:  # > 1ms
            opportunities.append((name, t))
    for name, t in opportunities:
        print(f"    [{fmt_ms(t):>10s}] {name}")

    if abs(gap_from_e2e) > 0.005:  # > 5ms
        if gap_from_e2e > 0:
            print(f"    [{fmt_ms(gap_from_e2e):>10s}] UNACCOUNTED GAP (unmeasured work)")
        else:
            print(f"    [{fmt_ms(gap_from_e2e):>10s}] NEGATIVE GAP (measurement interference)")

    # Theoretical minimum
    print(f"\n  === THEORETICAL ANALYSIS ===")
    print(f"  Memory footprint during perm loop:")
    print(f"    X_T:          {_X_T.nbytes/1e6:>8.1f} MB (read once per batch)")
    print(f"    signs:        {_all_signs.nbytes/1e6:>8.1f} MB (read once)")
    print(f"    t_batch:      {_t_batch.nbytes/1e6:>8.1f} MB (write then read)")
    print(f"    work bufs:    {(_par_idx_bufs.nbytes + _par_flat_maps.nbytes + _par_sums_bufs.nbytes)/1e6:>8.1f} MB")
    print(f"    sum_sq:       {_sum_sq.nbytes/1e6:>8.1f} MB (read once per batch)")

    # Bandwidth analysis for ttest
    bytes_per_batch_ttest = (_X_T.nbytes + BATCH_SIZE * n_samp * 8 +
                             BATCH_SIZE * n_tests * 8 + n_tests * 8)
    median_ttest = np.median(perm_results["batch_ttest"])
    bw_ttest = (bytes_per_batch_ttest / 1e9) / median_ttest
    print(f"\n  Bandwidth utilization (ttest):")
    print(f"    Data per batch:   {bytes_per_batch_ttest/1e6:.1f} MB")
    print(f"    Achieved BW:      {bw_ttest:.1f} GB/s")
    print(f"    (typical DRAM: 25-50 GB/s for desktop, 100+ for server)")

    print()


if __name__ == "__main__":
    main()
