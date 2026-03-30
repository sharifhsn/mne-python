"""v9 HPC benchmark: batched fused ttest end-to-end.

A/B comparison: v9 (batched _batched_fused_ttest, reads X_T once per batch)
vs v8 (unbatched, reads X_T once per perm — simulated by monkeypatching
_batched_fused_ttest to call _fused_ttest in a loop).

Memory-optimized: uses out_type="indices" and frees results between phases.
"""

import gc
import time
import numpy as np
import mne
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.fixes import has_numba
import mne.stats.cluster_level as cl


# Save originals at import time
_orig_batched = cl._batched_fused_ttest
_orig_fused = cl._fused_ttest


def unbatched_ttest(X_T, signs_batch, sum_sq, inv_n, neg_n, sqrt_n_nm1,
                    t_batch, n_batch):
    """Call _fused_ttest individually for each perm (simulates v8 behavior).

    Reads X_T n_batch times (once per perm) instead of once for the whole
    batch.  This is the performance baseline we're comparing against.
    """
    for b in range(n_batch):
        _orig_fused(X_T, signs_batch[b], sum_sq, inv_n, neg_n, sqrt_n_nm1,
                    t_batch[b])


def _set_v8():
    """Monkeypatch to unbatched (v8) behavior."""
    cl._batched_fused_ttest = unbatched_ttest


def _set_v9():
    """Restore batched (v9) behavior."""
    cl._batched_fused_ttest = _orig_batched


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
    # v9 — Warmup (JIT compilation for batched kernel)
    # ==================================================================
    print("\n--- v9 (batched fused ttest, B=32) ---")
    print("  Warming up JIT...", end=" ", flush=True)
    _r = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=16, threshold=threshold,
        tail=1, seed=42, verbose=False, out_type="indices",
    )
    del _r; gc.collect()
    print("done.")

    # ==================================================================
    # v9 — Timed runs
    # ==================================================================
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
    n_clusters_v9 = len(clusters_v9)
    print(f"  n_clusters: {n_clusters_v9}, "
          f"t_obs range: [{t_obs_v9.min():.3f}, {t_obs_v9.max():.3f}]")
    if len(pvals_v9) > 0:
        n_sig = np.sum(pvals_v9 < 0.05)
        print(f"  p-values: [{pvals_v9.min():.4f}, {pvals_v9.max():.4f}], "
              f"{n_sig} significant (p<0.05)")
    print(f"  H0 range: [{H0_v9.min():.3f}, {H0_v9.max():.3f}]")

    # Determinism check
    t_obs_v9b, _, pvals_v9b, H0_v9b = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=n_permutations,
        threshold=threshold, tail=1, seed=42, verbose=False,
        out_type="indices",
    )
    assert np.array_equal(t_obs_v9, t_obs_v9b), "t_obs not deterministic!"
    assert np.array_equal(H0_v9, H0_v9b), "H0 not deterministic!"
    assert np.array_equal(pvals_v9, pvals_v9b), "pvals not deterministic!"
    print("  Determinism: PASS")
    del t_obs_v9b, pvals_v9b, H0_v9b; gc.collect()

    # ==================================================================
    # v8 — Warmup + Timed runs (unbatched, monkeypatched)
    # ==================================================================
    print("\n--- v8 (unbatched fused ttest, monkeypatched) ---")
    _set_v8()

    print("  Warming up...", end=" ", flush=True)
    _r = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=16, threshold=threshold,
        tail=1, seed=42, verbose=False, out_type="indices",
    )
    del _r; gc.collect()
    print("done.")

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
        if run == 4:
            t_obs_v8, clusters_v8, pvals_v8, H0_v8 = _r
        del _r; gc.collect()

    v8_median = np.median(v8_times)
    print(f"  Times: {['%.3f' % t for t in v8_times]}")
    print(f"  MEDIAN: {v8_median:.3f}s  "
          f"({v8_median/n_permutations*1e3:.3f} ms/perm)")

    _set_v9()

    # ==================================================================
    # Parity check (v9 vs v8)
    # ==================================================================
    print(f"\n{'='*70}")
    print("PARITY CHECK (v9 batched vs v8 unbatched)")
    print(f"{'='*70}")
    t_obs_match = np.allclose(t_obs_v9, t_obs_v8, atol=1e-10)
    H0_match = np.allclose(H0_v9, H0_v8, atol=1e-10)
    pvals_match = np.allclose(pvals_v9, pvals_v8, atol=1e-10)
    n_clusters_v8 = len(clusters_v8)
    n_clusters_match = n_clusters_v9 == n_clusters_v8

    print(f"  t_obs match:      {t_obs_match}")
    if not t_obs_match:
        diff = np.abs(t_obs_v9 - t_obs_v8)
        print(f"    max diff: {diff.max():.2e}, "
              f"mismatches: {np.sum(diff > 1e-10)}")
    print(f"  H0 match:         {H0_match}")
    if not H0_match:
        diff = np.abs(H0_v9 - H0_v8)
        print(f"    max diff: {diff.max():.2e}, "
              f"mismatches: {np.sum(diff > 1e-10)}")
    print(f"  p-values match:   {pvals_match}")
    if not pvals_match:
        diff = np.abs(pvals_v9 - pvals_v8)
        print(f"    max diff: {diff.max():.2e}, "
              f"mismatches: {np.sum(diff > 1e-10)}")
    print(f"  n_clusters match: {n_clusters_match} "
          f"({n_clusters_v9} vs {n_clusters_v8})")

    all_match = t_obs_match and H0_match and pvals_match and n_clusters_match
    print(f"\n  Overall parity: {'PASS' if all_match else 'FAIL'}")

    del t_obs_v9, clusters_v9, pvals_v9, H0_v9
    del t_obs_v8, clusters_v8, pvals_v8, H0_v8
    gc.collect()

    # ==================================================================
    # Multi-tail test
    # ==================================================================
    print(f"\n{'='*70}")
    print("MULTI-TAIL PARITY + PERF")
    print(f"{'='*70}")

    for tail, thresh in [(0, 1.67), (-1, -1.67)]:
        print(f"\n  tail={tail}, threshold={thresh}")

        # v9
        t0 = time.perf_counter()
        t_v9, cl_v9, p_v9, h_v9 = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_permutations,
            threshold=thresh, tail=tail, seed=42, verbose=False,
            out_type="indices",
        )
        t_v9_time = time.perf_counter() - t0

        # v8
        _set_v8()
        t0 = time.perf_counter()
        t_v8, cl_v8, p_v8, h_v8 = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_permutations,
            threshold=thresh, tail=tail, seed=42, verbose=False,
            out_type="indices",
        )
        t_v8_time = time.perf_counter() - t0
        _set_v9()

        t_match = np.allclose(t_v9, t_v8, atol=1e-10)
        h_match = np.allclose(h_v9, h_v8, atol=1e-10)
        p_match = np.allclose(p_v9, p_v8, atol=1e-10)
        print(f"    v9: {t_v9_time:.3f}s, v8: {t_v8_time:.3f}s, "
              f"speedup: {t_v8_time/t_v9_time:.2f}x")
        print(f"    parity: t_obs={t_match}, H0={h_match}, pvals={p_match}")
        if not (t_match and h_match and p_match):
            if not h_match:
                diff = np.abs(h_v9 - h_v8)
                print(f"    H0 max diff: {diff.max():.2e}")

        del t_v9, cl_v9, p_v9, h_v9, t_v8, cl_v8, p_v8, h_v8
        gc.collect()

    # ==================================================================
    # Scaling test
    # ==================================================================
    print(f"\n{'='*70}")
    print("SCALING (tail=1)")
    print(f"{'='*70}")

    for n_p in [256, 512, 1024, 2048, 4096]:
        # v9
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        t_v9_s = time.perf_counter() - t0
        del _r; gc.collect()

        # v8
        _set_v8()
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        t_v8_s = time.perf_counter() - t0
        _set_v9()
        del _r; gc.collect()

        print(f"  {n_p:5d} perms: v9={t_v9_s:.3f}s  v8={t_v8_s:.3f}s  "
              f"speedup={t_v8_s/t_v9_s:.2f}x  "
              f"({t_v9_s/n_p*1e3:.3f} vs {t_v8_s/n_p*1e3:.3f} ms/perm)")

    # Linear regression for per-perm cost
    times_v9_lr, times_v8_lr, perms_lr = [], [], []
    for n_p in [512, 1024, 2048, 4096]:
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        times_v9_lr.append(time.perf_counter() - t0)
        del _r; gc.collect()

        _set_v8()
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        times_v8_lr.append(time.perf_counter() - t0)
        _set_v9()
        del _r; gc.collect()

        perms_lr.append(n_p)

    A = np.vstack([perms_lr, np.ones(len(perms_lr))]).T
    slope_v9, intercept_v9 = np.linalg.lstsq(A, times_v9_lr, rcond=None)[0]
    slope_v8, intercept_v8 = np.linalg.lstsq(A, times_v8_lr, rcond=None)[0]
    print(f"\n  v9 linear: overhead={intercept_v9*1e3:.1f}ms, "
          f"per_perm={slope_v9*1e3:.3f}ms")
    print(f"  v8 linear: overhead={intercept_v8*1e3:.1f}ms, "
          f"per_perm={slope_v8*1e3:.3f}ms")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print("SUMMARY - v9 Batched Fused Ttest")
    print(f"{'='*70}")
    print(f"Data: {n_subjects} subjects x {n_src} verts x {n_times} times "
          f"= {n_src*n_times:,} tests")
    print(f"Adjacency: fsaverage ico-5 ({n_src} vertices)")
    print()
    print(f"v9 (batched B=32): {v9_median:.3f}s  "
          f"({v9_median/n_permutations*1e3:.3f} ms/perm)")
    print(f"  Linear: overhead={intercept_v9*1e3:.1f}ms, "
          f"per_perm={slope_v9*1e3:.3f}ms")
    print(f"v8 (unbatched):    {v8_median:.3f}s  "
          f"({v8_median/n_permutations*1e3:.3f} ms/perm)")
    print(f"  Linear: overhead={intercept_v8*1e3:.1f}ms, "
          f"per_perm={slope_v8*1e3:.3f}ms")
    print()
    print(f"End-to-end speedup ({n_permutations} perms): "
          f"{v8_median/v9_median:.2f}x")
    print(f"Per-perm speedup (linear slope): "
          f"{slope_v8/slope_v9:.2f}x")
    print(f"Parity: {'ALL PASS' if all_match else 'FAIL'}")
    print()
    print(f"Projected at 5000 perms:")
    t_v9_5k = intercept_v9 + 5000 * slope_v9
    t_v8_5k = intercept_v8 + 5000 * slope_v8
    print(f"  v9: {t_v9_5k:.1f}s")
    print(f"  v8: {t_v8_5k:.1f}s")
    print(f"  Savings: {t_v8_5k - t_v9_5k:.1f}s "
          f"({(1-t_v9_5k/t_v8_5k)*100:.0f}% faster)")


if __name__ == "__main__":
    run_benchmark()
