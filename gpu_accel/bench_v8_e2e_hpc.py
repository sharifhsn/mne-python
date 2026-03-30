"""v8 HPC benchmark: full spatio_temporal_cluster_1samp_test end-to-end.

Tests the ACTUAL researcher workflow, not just the inner permutation loop.
Includes: data setup, initial clustering, permutation loop, p-value computation.

A/B comparison via monkeypatching _fused_ttest with a numpy equivalent
to simulate v7 behavior through the same code path.

Memory-optimized: uses out_type="indices" and frees results between phases.
"""

import gc
import time
import numpy as np
import mne
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.fixes import has_numba
import mne.stats.cluster_level as cl


def numpy_ttest(X_T, signs, sum_sq, inv_n, neg_n, sqrt_n_nm1, t_buf):
    """NumPy version (v7 behavior): dot + 8 elementwise ops."""
    np.dot(signs, X_T.T, out=t_buf)
    t_buf *= inv_n  # t_buf = mean_s
    # Need temp for denom
    denom_sq = t_buf * t_buf
    denom_sq *= neg_n
    denom_sq += sum_sq
    np.maximum(denom_sq, 0, out=denom_sq)
    np.sqrt(denom_sq, out=denom_sq)
    np.divide(t_buf, denom_sq, out=t_buf)
    t_buf *= sqrt_n_nm1
    t_buf[~np.isfinite(t_buf)] = 0.0


def numpy_threshold_to_indices(t_buf, threshold, idx_buf):
    """NumPy version: np.where fallback."""
    idx = np.where(t_buf > threshold)[0].astype(np.intp)
    n = len(idx)
    idx_buf[:n] = idx
    return n


def numpy_threshold_to_indices_neg(t_buf, threshold, idx_buf):
    """NumPy version: np.where fallback (negative tail)."""
    idx = np.where(t_buf < threshold)[0].astype(np.intp)
    n = len(idx)
    idx_buf[:n] = idx
    return n


def _set_v7():
    """Monkeypatch to numpy versions."""
    cl._fused_ttest = numpy_ttest
    cl._threshold_to_indices = numpy_threshold_to_indices
    cl._threshold_to_indices_neg = numpy_threshold_to_indices_neg


def _set_v8(orig):
    """Restore fused Numba versions."""
    cl._fused_ttest = orig["fused"]
    cl._threshold_to_indices = orig["thresh"]
    cl._threshold_to_indices_neg = orig["thresh_neg"]


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

    # Save originals for monkeypatching
    orig = {
        "fused": cl._fused_ttest,
        "thresh": cl._threshold_to_indices,
        "thresh_neg": cl._threshold_to_indices_neg,
    }

    # ==================================================================
    # v8 — Warmup (JIT compilation)
    # ==================================================================
    print("\n--- v8 (fused Numba ttest) — full spatio_temporal_cluster_1samp_test ---")
    print("  Warming up JIT...", end=" ", flush=True)
    _r = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=16, threshold=threshold,
        tail=1, seed=42, verbose=False, out_type="indices",
    )
    del _r; gc.collect()
    print("done.")

    # ==================================================================
    # v8 — Timed runs (discard results, keep only timing)
    # ==================================================================
    v8_times = []
    for run in range(5):
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_permutations,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        t1 = time.perf_counter()
        v8_times.append(t1 - t0)
        # Keep last run for stats reporting
        if run == 4:
            t_obs_v8, clusters_v8, pvals_v8, H0_v8 = _r
        del _r; gc.collect()

    v8_median = np.median(v8_times)
    print(f"  Times: {['%.3f' % t for t in v8_times]}")
    print(f"  MEDIAN: {v8_median:.3f}s  "
          f"({v8_median/n_permutations*1e3:.3f} ms/perm)")
    n_clusters_v8 = len(clusters_v8)
    print(f"  n_clusters: {n_clusters_v8}, "
          f"t_obs range: [{t_obs_v8.min():.3f}, {t_obs_v8.max():.3f}]")
    if len(pvals_v8) > 0:
        n_sig = np.sum(pvals_v8 < 0.05)
        print(f"  p-values: [{pvals_v8.min():.4f}, {pvals_v8.max():.4f}], "
              f"{n_sig} significant (p<0.05)")
    print(f"  H0 range: [{H0_v8.min():.3f}, {H0_v8.max():.3f}]")

    # Determinism check (separate run, same seed)
    t_obs_v8b, _, pvals_v8b, H0_v8b = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=n_permutations,
        threshold=threshold, tail=1, seed=42, verbose=False,
        out_type="indices",
    )
    assert np.array_equal(t_obs_v8, t_obs_v8b), "t_obs not deterministic!"
    assert np.array_equal(H0_v8, H0_v8b), "H0 not deterministic!"
    assert np.array_equal(pvals_v8, pvals_v8b), "pvals not deterministic!"
    print("  Determinism: PASS")
    del t_obs_v8b, pvals_v8b, H0_v8b; gc.collect()

    # ==================================================================
    # v7 — Warmup + Timed runs
    # ==================================================================
    print("\n--- v7 (numpy ttest, monkeypatched) — full spatio_temporal_cluster_1samp_test ---")
    _set_v7()

    print("  Warming up...", end=" ", flush=True)
    _r = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=16, threshold=threshold,
        tail=1, seed=42, verbose=False, out_type="indices",
    )
    del _r; gc.collect()
    print("done.")

    v7_times = []
    for run in range(5):
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_permutations,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        t1 = time.perf_counter()
        v7_times.append(t1 - t0)
        if run == 4:
            t_obs_v7, clusters_v7, pvals_v7, H0_v7 = _r
        del _r; gc.collect()

    v7_median = np.median(v7_times)
    print(f"  Times: {['%.3f' % t for t in v7_times]}")
    print(f"  MEDIAN: {v7_median:.3f}s  "
          f"({v7_median/n_permutations*1e3:.3f} ms/perm)")

    _set_v8(orig)

    # ==================================================================
    # Parity check (v8 vs v7 from final timed runs)
    # ==================================================================
    print(f"\n{'='*70}")
    print("PARITY CHECK")
    print(f"{'='*70}")
    t_obs_match = np.allclose(t_obs_v8, t_obs_v7, atol=1e-10)
    H0_match = np.allclose(H0_v8, H0_v7, atol=1e-10)
    pvals_match = np.allclose(pvals_v8, pvals_v7, atol=1e-10)
    n_clusters_v7 = len(clusters_v7)
    n_clusters_match = n_clusters_v8 == n_clusters_v7

    print(f"  t_obs match:      {t_obs_match}")
    if not t_obs_match:
        diff = np.abs(t_obs_v8 - t_obs_v7)
        print(f"    max diff: {diff.max():.2e}, mismatches: {np.sum(diff > 1e-10)}")
    print(f"  H0 match:         {H0_match}")
    if not H0_match:
        diff = np.abs(H0_v8 - H0_v7)
        print(f"    max diff: {diff.max():.2e}, mismatches: {np.sum(diff > 1e-10)}")
    print(f"  p-values match:   {pvals_match}")
    if not pvals_match:
        diff = np.abs(pvals_v8 - pvals_v7)
        print(f"    max diff: {diff.max():.2e}, mismatches: {np.sum(diff > 1e-10)}")
    print(f"  n_clusters match: {n_clusters_match} "
          f"({n_clusters_v8} vs {n_clusters_v7})")

    all_match = t_obs_match and H0_match and pvals_match and n_clusters_match
    print(f"\n  Overall parity: {'PASS' if all_match else 'FAIL'}")

    # Free parity data
    del t_obs_v8, clusters_v8, pvals_v8, H0_v8
    del t_obs_v7, clusters_v7, pvals_v7, H0_v7
    gc.collect()

    # ==================================================================
    # Multi-tail test (tail=0, tail=-1)
    # ==================================================================
    print(f"\n{'='*70}")
    print("MULTI-TAIL PARITY + PERF")
    print(f"{'='*70}")

    for tail, thresh in [(0, 1.67), (-1, -1.67)]:
        print(f"\n  tail={tail}, threshold={thresh}")

        # v8
        t0 = time.perf_counter()
        t_v8, cl_v8, p_v8, h_v8 = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_permutations,
            threshold=thresh, tail=tail, seed=42, verbose=False,
            out_type="indices",
        )
        t_v8_time = time.perf_counter() - t0

        # v7
        _set_v7()
        t0 = time.perf_counter()
        t_v7, cl_v7, p_v7, h_v7 = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_permutations,
            threshold=thresh, tail=tail, seed=42, verbose=False,
            out_type="indices",
        )
        t_v7_time = time.perf_counter() - t0
        _set_v8(orig)

        t_match = np.allclose(t_v8, t_v7, atol=1e-10)
        h_match = np.allclose(h_v8, h_v7, atol=1e-10)
        p_match = np.allclose(p_v8, p_v7, atol=1e-10)
        print(f"    v8: {t_v8_time:.3f}s, v7: {t_v7_time:.3f}s, "
              f"speedup: {t_v7_time/t_v8_time:.2f}x")
        print(f"    parity: t_obs={t_match}, H0={h_match}, pvals={p_match}")
        if not (t_match and h_match and p_match):
            if not h_match:
                diff = np.abs(h_v8 - h_v7)
                print(f"    H0 max diff: {diff.max():.2e}")

        del t_v8, cl_v8, p_v8, h_v8, t_v7, cl_v7, p_v7, h_v7
        gc.collect()

    # ==================================================================
    # Scaling test: varying n_permutations
    # ==================================================================
    print(f"\n{'='*70}")
    print("SCALING (tail=1)")
    print(f"{'='*70}")

    for n_p in [256, 512, 1024, 2048, 4096]:
        # v8
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        t_v8_s = time.perf_counter() - t0
        del _r; gc.collect()

        # v7
        _set_v7()
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        t_v7_s = time.perf_counter() - t0
        _set_v8(orig)
        del _r; gc.collect()

        print(f"  {n_p:5d} perms: v8={t_v8_s:.3f}s  v7={t_v7_s:.3f}s  "
              f"speedup={t_v7_s/t_v8_s:.2f}x  "
              f"({t_v8_s/n_p*1e3:.3f} vs {t_v7_s/n_p*1e3:.3f} ms/perm)")

    # Linear regression for per-perm cost
    times_v8_lr, times_v7_lr, perms_lr = [], [], []
    for n_p in [512, 1024, 2048, 4096]:
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        times_v8_lr.append(time.perf_counter() - t0)
        del _r; gc.collect()

        _set_v7()
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        times_v7_lr.append(time.perf_counter() - t0)
        _set_v8(orig)
        del _r; gc.collect()

        perms_lr.append(n_p)

    A = np.vstack([perms_lr, np.ones(len(perms_lr))]).T
    slope_v8, intercept_v8 = np.linalg.lstsq(A, times_v8_lr, rcond=None)[0]
    slope_v7, intercept_v7 = np.linalg.lstsq(A, times_v7_lr, rcond=None)[0]
    print(f"\n  v8 linear: overhead={intercept_v8*1e3:.1f}ms, "
          f"per_perm={slope_v8*1e3:.3f}ms")
    print(f"  v7 linear: overhead={intercept_v7*1e3:.1f}ms, "
          f"per_perm={slope_v7*1e3:.3f}ms")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print("SUMMARY - Full spatio_temporal_cluster_1samp_test")
    print(f"{'='*70}")
    print(f"Data: {n_subjects} subjects x {n_src} verts x {n_times} times "
          f"= {n_src*n_times:,} tests")
    print(f"Adjacency: fsaverage ico-5 ({n_src} vertices)")
    print()
    print(f"v8 (fused Numba): {v8_median:.3f}s  "
          f"({v8_median/n_permutations*1e3:.3f} ms/perm)")
    print(f"  Linear: overhead={intercept_v8*1e3:.1f}ms, "
          f"per_perm={slope_v8*1e3:.3f}ms")
    print(f"v7 (numpy):       {v7_median:.3f}s  "
          f"({v7_median/n_permutations*1e3:.3f} ms/perm)")
    print(f"  Linear: overhead={intercept_v7*1e3:.1f}ms, "
          f"per_perm={slope_v7*1e3:.3f}ms")
    print()
    print(f"End-to-end speedup ({n_permutations} perms): "
          f"{v7_median/v8_median:.2f}x")
    print(f"Per-perm speedup (linear slope): "
          f"{slope_v7/slope_v8:.2f}x")
    print(f"Parity: {'ALL PASS' if all_match else 'FAIL'}")
    print()
    print(f"Projected at 5000 perms:")
    t_v8_5k = intercept_v8 + 5000 * slope_v8
    t_v7_5k = intercept_v7 + 5000 * slope_v7
    print(f"  v8: {t_v8_5k:.1f}s")
    print(f"  v7: {t_v7_5k:.1f}s")
    print(f"  Savings: {t_v7_5k - t_v8_5k:.1f}s ({(1-t_v8_5k/t_v7_5k)*100:.0f}% faster)")


if __name__ == "__main__":
    run_benchmark()
