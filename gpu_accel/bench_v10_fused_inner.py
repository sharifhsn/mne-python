"""v10 HPC benchmark: fused Numba inner loop end-to-end.

A/B comparison: v10 (_perm_batch_fast fuses threshold + CCL + bincount +
argmax into a single Numba call) vs v9 (Python for-loop calling individual
Numba kernels — simulated by disabling _use_fused_inner).
"""

import gc
import time
import numpy as np
import mne
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.fixes import has_numba
import mne.stats.cluster_level as cl


# Save the original at import time
_orig_perm_batch_fast = cl._perm_batch_fast
_orig_threshold = cl._threshold_to_indices
_orig_threshold_neg = cl._threshold_to_indices_neg
_orig_st_ccl = cl._st_fused_ccl


def _python_loop_inner(
    t_batch, n_batch, batch_start,
    threshold, tail, t_power,
    idx_buf, flat_map, indptr, indices, n_src, max_step,
    max_cluster_sums, sums_buf,
):
    """Python loop version (simulates v9 behavior).

    Calls individual Numba kernels from a Python for-loop, reproducing
    the exact per-perm overhead that _perm_batch_fast eliminates.
    """
    for b in range(n_batch):
        seed_idx = batch_start + b
        best = 0.0
        tail_specs = []
        if tail == 0:
            tail_specs = [
                (threshold, _orig_threshold),
                (-threshold, _orig_threshold_neg),
            ]
        elif tail == 1:
            tail_specs = [(threshold, _orig_threshold)]
        else:
            tail_specs = [(threshold, _orig_threshold_neg)]

        for thresh, thresh_fn in tail_specs:
            n_act = thresh_fn(t_batch[b], thresh, idx_buf)
            act_idx = idx_buf[:n_act]
            if n_act == 0:
                continue
            comps = _orig_st_ccl(
                act_idx, n_act, flat_map,
                indptr, indices, n_src, max_step,
            )
            if t_power == 1:
                sums = np.bincount(comps, weights=t_batch[b][act_idx])
            else:
                vals = (
                    np.sign(t_batch[b][act_idx])
                    * np.abs(t_batch[b][act_idx]) ** t_power
                )
                sums = np.bincount(comps, weights=vals)
            idx_max = np.argmax(np.abs(sums))
            if abs(sums[idx_max]) > abs(best):
                best = sums[idx_max]

        max_cluster_sums[seed_idx] = best


def _set_v9():
    """Replace fused inner with Python loop (v9 behavior)."""
    cl._perm_batch_fast = _python_loop_inner


def _set_v10():
    """Restore fused Numba inner loop."""
    cl._perm_batch_fast = _orig_perm_batch_fast


def run_benchmark():
    # ---- Setup real fsaverage data ----
    data_path = mne.datasets.sample.data_path()
    subjects_dir = data_path / "subjects"
    src = mne.read_source_spaces(
        subjects_dir / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif",
        verbose=False,
    )
    adjacency = mne.spatial_src_adjacency(src, verbose=False)
    n_src = adjacency.shape[0]
    n_subjects = 15
    n_times = 15

    # Create realistic test data
    np.random.seed(123)
    X = np.random.randn(n_subjects, n_times, n_src) * 0.3
    # Add signal: 100 vertices × 5 timepoints
    signal_verts = np.arange(100, 200)
    for t in range(5, 10):
        X[:, t, signal_verts] += 2.0

    threshold = 1.67
    n_permutations = 2048

    print(f"Setup: {n_subjects} subjects × {n_src} vertices × {n_times} times")
    print(f"       = {n_src * n_times:,} tests, {n_permutations} permutations")
    print(f"has_numba = {has_numba}")
    print(f"threshold = {threshold}")
    print("=" * 70)

    # ==================================================================
    # v10 — Warmup (JIT compilation for all kernels)
    # ==================================================================
    print("\n--- v10 (fused Numba inner loop) ---")
    print("  Warming up JIT...", end=" ", flush=True)
    _r = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=16, threshold=threshold,
        tail=1, seed=42, verbose=False, out_type="indices",
    )
    del _r; gc.collect()
    print("done.")

    # ==================================================================
    # v10 — Timed runs
    # ==================================================================
    v10_times = []
    for run in range(5):
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_permutations,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        t1 = time.perf_counter()
        v10_times.append(t1 - t0)
        if run == 4:
            t_obs_v10, clusters_v10, pvals_v10, H0_v10 = _r
        del _r; gc.collect()

    v10_median = np.median(v10_times)
    print(f"  Times: {['%.3f' % t for t in v10_times]}")
    print(f"  MEDIAN: {v10_median:.3f}s  "
          f"({v10_median/n_permutations*1e3:.3f} ms/perm)")
    n_clusters_v10 = len(clusters_v10)
    print(f"  n_clusters: {n_clusters_v10}, "
          f"t_obs range: [{t_obs_v10.min():.3f}, {t_obs_v10.max():.3f}]")
    if len(pvals_v10) > 0:
        n_sig = np.sum(pvals_v10 < 0.05)
        print(f"  p-values: [{pvals_v10.min():.4f}, {pvals_v10.max():.4f}], "
              f"{n_sig} significant (p<0.05)")
    print(f"  H0 range: [{H0_v10.min():.3f}, {H0_v10.max():.3f}]")

    # Determinism check
    t_obs_10b, _, pvals_10b, H0_10b = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=n_permutations,
        threshold=threshold, tail=1, seed=42, verbose=False,
        out_type="indices",
    )
    assert np.array_equal(t_obs_v10, t_obs_10b), "t_obs not deterministic!"
    assert np.array_equal(H0_v10, H0_10b), "H0 not deterministic!"
    assert np.array_equal(pvals_v10, pvals_10b), "pvals not deterministic!"
    print("  Determinism: PASS")
    del t_obs_10b, pvals_10b, H0_10b; gc.collect()

    # ==================================================================
    # v9 — Warmup + Timed runs (Python loop, _perm_batch_fast disabled)
    # ==================================================================
    print("\n--- v9 (Python loop, fused inner disabled) ---")
    _set_v9()

    print("  Warming up...", end=" ", flush=True)
    _r = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=16, threshold=threshold,
        tail=1, seed=42, verbose=False, out_type="indices",
    )
    del _r; gc.collect()
    print("done.")

    v9_times = []
    for run in range(5):
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_permutations,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        t1 = time.perf_counter()
        v9_times.append(t1 - t0)
        if run == 4:
            t_obs_v9, clusters_v9, pvals_v9, H0_v9 = _r
        del _r; gc.collect()

    v9_median = np.median(v9_times)
    print(f"  Times: {['%.3f' % t for t in v9_times]}")
    print(f"  MEDIAN: {v9_median:.3f}s  "
          f"({v9_median/n_permutations*1e3:.3f} ms/perm)")

    _set_v10()

    # ==================================================================
    # Parity check (v10 vs v9)
    # ==================================================================
    print(f"\n{'='*70}")
    print("PARITY CHECK (v10 fused Numba vs v9 Python loop)")
    print(f"{'='*70}")
    t_obs_match = np.allclose(t_obs_v10, t_obs_v9, atol=1e-10)
    H0_match = np.allclose(H0_v10, H0_v9, atol=1e-10)
    pvals_match = np.allclose(pvals_v10, pvals_v9, atol=1e-10)
    n_clusters_v9 = len(clusters_v9)
    n_clusters_match = n_clusters_v10 == n_clusters_v9

    print(f"  t_obs match:      {t_obs_match}")
    if not t_obs_match:
        diff = np.abs(t_obs_v10 - t_obs_v9)
        print(f"    max diff: {diff.max():.2e}, "
              f"mismatches: {np.sum(diff > 1e-10)}")
    print(f"  H0 match:         {H0_match}")
    if not H0_match:
        diff = np.abs(H0_v10 - H0_v9)
        print(f"    max diff: {diff.max():.2e}, "
              f"mismatches: {np.sum(diff > 1e-10)}")
    print(f"  p-values match:   {pvals_match}")
    if not pvals_match:
        diff = np.abs(pvals_v10 - pvals_v9)
        print(f"    max diff: {diff.max():.2e}, "
              f"mismatches: {np.sum(diff > 1e-10)}")
    print(f"  n_clusters match: {n_clusters_match} "
          f"({n_clusters_v10} vs {n_clusters_v9})")

    all_match = t_obs_match and H0_match and pvals_match and n_clusters_match
    print(f"\n  Overall parity: {'PASS' if all_match else 'FAIL'}")

    del t_obs_v10, clusters_v10, pvals_v10, H0_v10
    del t_obs_v9, clusters_v9, pvals_v9, H0_v9
    gc.collect()

    # ==================================================================
    # Multi-tail test
    # ==================================================================
    print(f"\n{'='*70}")
    print("MULTI-TAIL PARITY + PERF")
    print(f"{'='*70}")

    for tail, thresh in [(0, 1.67), (-1, -1.67)]:
        print(f"\n  tail={tail}, threshold={thresh}")

        # v10
        t0 = time.perf_counter()
        t_v10, cl_v10, p_v10, h_v10 = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_permutations,
            threshold=thresh, tail=tail, seed=42, verbose=False,
            out_type="indices",
        )
        t_v10_time = time.perf_counter() - t0

        # v9
        _set_v9()
        t0 = time.perf_counter()
        t_v9, cl_v9, p_v9, h_v9 = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_permutations,
            threshold=thresh, tail=tail, seed=42, verbose=False,
            out_type="indices",
        )
        t_v9_time = time.perf_counter() - t0
        _set_v10()

        t_match = np.allclose(t_v10, t_v9, atol=1e-10)
        h_match = np.allclose(h_v10, h_v9, atol=1e-10)
        p_match = np.allclose(p_v10, p_v9, atol=1e-10)
        print(f"    v10: {t_v10_time:.3f}s, v9: {t_v9_time:.3f}s, "
              f"speedup: {t_v9_time/t_v10_time:.2f}x")
        print(f"    parity: t_obs={t_match}, H0={h_match}, pvals={p_match}")
        if not (t_match and h_match and p_match):
            if not h_match:
                diff = np.abs(h_v10 - h_v9)
                print(f"    H0 max diff: {diff.max():.2e}")

        del t_v10, cl_v10, p_v10, h_v10, t_v9, cl_v9, p_v9, h_v9
        gc.collect()

    # ==================================================================
    # Scaling test
    # ==================================================================
    print(f"\n{'='*70}")
    print("SCALING (tail=1)")
    print(f"{'='*70}")

    for n_p in [256, 512, 1024, 2048, 4096]:
        # v10
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        t_v10_s = time.perf_counter() - t0
        del _r; gc.collect()

        # v9
        _set_v9()
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        t_v9_s = time.perf_counter() - t0
        _set_v10()
        del _r; gc.collect()

        print(f"  {n_p:5d} perms: v10={t_v10_s:.3f}s  v9={t_v9_s:.3f}s  "
              f"speedup={t_v9_s/t_v10_s:.2f}x  "
              f"({t_v10_s/n_p*1e3:.3f} vs {t_v9_s/n_p*1e3:.3f} ms/perm)")

    # Linear regression for per-perm cost
    times_v10_lr, times_v9_lr, perms_lr = [], [], []
    for n_p in [512, 1024, 2048, 4096]:
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        times_v10_lr.append(time.perf_counter() - t0)
        del _r; gc.collect()

        _set_v9()
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        times_v9_lr.append(time.perf_counter() - t0)
        _set_v10()
        del _r; gc.collect()

        perms_lr.append(n_p)

    A = np.vstack([perms_lr, np.ones(len(perms_lr))]).T
    slope_v10, intercept_v10 = np.linalg.lstsq(A, times_v10_lr, rcond=None)[0]
    slope_v9, intercept_v9 = np.linalg.lstsq(A, times_v9_lr, rcond=None)[0]
    print(f"\n  v10 linear: overhead={intercept_v10*1e3:.1f}ms, "
          f"per_perm={slope_v10*1e3:.3f}ms")
    print(f"  v9  linear: overhead={intercept_v9*1e3:.1f}ms, "
          f"per_perm={slope_v9*1e3:.3f}ms")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print("SUMMARY - v10 Fused Numba Inner Loop")
    print(f"{'='*70}")
    print(f"Data: {n_subjects} subjects x {n_src} verts x {n_times} times "
          f"= {n_src*n_times:,} tests")
    print(f"Adjacency: fsaverage ico-5 ({n_src} vertices)")
    print()
    print(f"v10 (fused Numba inner): {v10_median:.3f}s  "
          f"({v10_median/n_permutations*1e3:.3f} ms/perm)")
    print(f"  Linear: overhead={intercept_v10*1e3:.1f}ms, "
          f"per_perm={slope_v10*1e3:.3f}ms")
    print(f"v9  (Python loop):       {v9_median:.3f}s  "
          f"({v9_median/n_permutations*1e3:.3f} ms/perm)")
    print(f"  Linear: overhead={intercept_v9*1e3:.1f}ms, "
          f"per_perm={slope_v9*1e3:.3f}ms")
    print()
    print(f"End-to-end speedup ({n_permutations} perms): "
          f"{v9_median/v10_median:.2f}x")
    print(f"Per-perm speedup (linear slope): "
          f"{slope_v9/slope_v10:.2f}x")
    print(f"Parity: {'ALL PASS' if all_match else 'FAIL'}")
    print()
    print(f"Projected at 5000 perms:")
    t_v10_5k = intercept_v10 + 5000 * slope_v10
    t_v9_5k = intercept_v9 + 5000 * slope_v9
    print(f"  v10: {t_v10_5k:.1f}s")
    print(f"  v9:  {t_v9_5k:.1f}s")
    print(f"  Savings: {t_v9_5k - t_v10_5k:.1f}s "
          f"({(1-t_v10_5k/t_v9_5k)*100:.0f}% faster)")


if __name__ == "__main__":
    run_benchmark()
