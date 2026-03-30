"""v11 HPC benchmark: parallel permutation processing via prange.

A/B comparison: v11 (_perm_batch_fast with prange(n_batch) parallelizes
CCL+threshold+bincount across perms) vs v10 (serial range(n_batch)).

The v10 baseline is simulated by monkeypatching _perm_batch_fast with a
serial @jit() wrapper that uses range() instead of prange().
"""

import gc
import time
import numpy as np
import mne
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.fixes import has_numba, jit
import mne.stats.cluster_level as cl


# Save the original (parallel prange version) at import time
_orig_perm_batch_fast = cl._perm_batch_fast
_orig_st_fused_ccl = cl._st_fused_ccl


@jit()
def _serial_perm_batch_fast(
    t_batch, n_batch, batch_start,
    threshold, tail, t_power,
    idx_bufs, flat_maps, indptr, indices, n_src, max_step,
    max_cluster_sums, sums_bufs,
):
    """Serial version of _perm_batch_fast (v10 behavior).

    Identical logic but uses range() instead of prange(),
    so the CCL runs on a single core.
    """
    n_vars = t_batch.shape[1]

    for b in range(n_batch):
        seed_idx = batch_start + b
        best = 0.0
        idx_buf = idx_bufs[b]
        flat_map = flat_maps[b]
        sums_buf = sums_bufs[b]

        for direction in range(2):
            if direction == 0 and tail == -1:
                continue
            if direction == 1 and tail == 1:
                continue

            n_act = 0
            if direction == 0:
                for j in range(n_vars):
                    if t_batch[b, j] > threshold:
                        idx_buf[n_act] = j
                        n_act += 1
            else:
                neg_thresh = -threshold if tail == 0 else threshold
                for j in range(n_vars):
                    if t_batch[b, j] < neg_thresh:
                        idx_buf[n_act] = j
                        n_act += 1

            if n_act == 0:
                continue

            comps = _orig_st_fused_ccl(
                idx_buf[:n_act], n_act, flat_map,
                indptr, indices, n_src, max_step,
            )

            max_comp = 0
            for k in range(n_act):
                if comps[k] > max_comp:
                    max_comp = comps[k]
            n_comps = max_comp + 1

            for c in range(n_comps):
                sums_buf[c] = 0.0

            if t_power == 1:
                for k in range(n_act):
                    sums_buf[comps[k]] += t_batch[b, idx_buf[k]]
            else:
                for k in range(n_act):
                    val = t_batch[b, idx_buf[k]]
                    if val >= 0.0:
                        sums_buf[comps[k]] += val ** t_power
                    else:
                        sums_buf[comps[k]] -= (-val) ** t_power

            dir_best = 0.0
            for c in range(n_comps):
                if abs(sums_buf[c]) > abs(dir_best):
                    dir_best = sums_buf[c]

            if abs(dir_best) > abs(best):
                best = dir_best

        max_cluster_sums[seed_idx] = best


def _set_v10():
    """Replace parallel prange with serial range (v10 behavior)."""
    cl._perm_batch_fast = _serial_perm_batch_fast


def _set_v11():
    """Restore parallel prange (v11 behavior)."""
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

    # Create random test data (worst-case for CCL: ~5% density, big clusters)
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
    # v11 — Warmup
    # ==================================================================
    print("\n--- v11 (parallel prange) ---")
    print("  Warming up JIT...", end=" ", flush=True)
    _r = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=16, threshold=threshold,
        tail=1, seed=42, verbose=False, out_type="indices",
    )
    del _r; gc.collect()
    print("done.")

    # ==================================================================
    # v11 — Timed runs
    # ==================================================================
    v11_times = []
    for run in range(5):
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_permutations,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        t1 = time.perf_counter()
        v11_times.append(t1 - t0)
        if run == 4:
            t_obs_v11, clusters_v11, pvals_v11, H0_v11 = _r
        del _r; gc.collect()

    v11_median = np.median(v11_times)
    print(f"  Times: {['%.3f' % t for t in v11_times]}")
    print(f"  MEDIAN: {v11_median:.3f}s  "
          f"({v11_median/n_permutations*1e3:.3f} ms/perm)")
    n_clusters_v11 = len(clusters_v11)
    print(f"  n_clusters: {n_clusters_v11}, "
          f"t_obs range: [{t_obs_v11.min():.3f}, {t_obs_v11.max():.3f}]")
    if len(pvals_v11) > 0:
        n_sig = np.sum(pvals_v11 < 0.05)
        print(f"  p-values: [{pvals_v11.min():.4f}, {pvals_v11.max():.4f}], "
              f"{n_sig} significant (p<0.05)")
    print(f"  H0 range: [{H0_v11.min():.3f}, {H0_v11.max():.3f}]")

    # Determinism check
    t_obs_11b, _, pvals_11b, H0_11b = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=n_permutations,
        threshold=threshold, tail=1, seed=42, verbose=False,
        out_type="indices",
    )
    assert np.array_equal(t_obs_v11, t_obs_11b), "t_obs not deterministic!"
    assert np.array_equal(H0_v11, H0_11b), "H0 not deterministic!"
    assert np.array_equal(pvals_v11, pvals_11b), "pvals not deterministic!"
    print("  Determinism: PASS")
    del t_obs_11b, pvals_11b, H0_11b; gc.collect()

    # ==================================================================
    # v10 — Warmup + Timed runs (serial range, prange disabled)
    # ==================================================================
    print("\n--- v10 (serial range) ---")
    _set_v10()

    print("  Warming up...", end=" ", flush=True)
    _r = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=16, threshold=threshold,
        tail=1, seed=42, verbose=False, out_type="indices",
    )
    del _r; gc.collect()
    print("done.")

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

    _set_v11()

    # ==================================================================
    # Parity check (v11 vs v10)
    # ==================================================================
    print(f"\n{'='*70}")
    print("PARITY CHECK (v11 parallel prange vs v10 serial range)")
    print(f"{'='*70}")
    t_obs_match = np.allclose(t_obs_v11, t_obs_v10, atol=1e-10)
    H0_match = np.allclose(H0_v11, H0_v10, atol=1e-10)
    pvals_match = np.allclose(pvals_v11, pvals_v10, atol=1e-10)
    n_clusters_v10 = len(clusters_v10)
    n_clusters_match = n_clusters_v11 == n_clusters_v10

    print(f"  t_obs match:      {t_obs_match}")
    if not t_obs_match:
        diff = np.abs(t_obs_v11 - t_obs_v10)
        print(f"    max diff: {diff.max():.2e}, "
              f"mismatches: {np.sum(diff > 1e-10)}")
    print(f"  H0 match:         {H0_match}")
    if not H0_match:
        diff = np.abs(H0_v11 - H0_v10)
        print(f"    max diff: {diff.max():.2e}, "
              f"mismatches: {np.sum(diff > 1e-10)}")
    print(f"  p-values match:   {pvals_match}")
    if not pvals_match:
        diff = np.abs(pvals_v11 - pvals_v10)
        print(f"    max diff: {diff.max():.2e}, "
              f"mismatches: {np.sum(diff > 1e-10)}")
    print(f"  n_clusters match: {n_clusters_match} "
          f"({n_clusters_v11} vs {n_clusters_v10})")

    all_match = t_obs_match and H0_match and pvals_match and n_clusters_match
    print(f"\n  Overall parity: {'PASS' if all_match else 'FAIL'}")

    del t_obs_v11, clusters_v11, pvals_v11, H0_v11
    del t_obs_v10, clusters_v10, pvals_v10, H0_v10
    gc.collect()

    # ==================================================================
    # Multi-tail test
    # ==================================================================
    print(f"\n{'='*70}")
    print("MULTI-TAIL PARITY + PERF")
    print(f"{'='*70}")

    for tail, thresh in [(0, 1.67), (-1, -1.67)]:
        print(f"\n  tail={tail}, threshold={thresh}")

        # v11
        t0 = time.perf_counter()
        t_v11, cl_v11, p_v11, h_v11 = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_permutations,
            threshold=thresh, tail=tail, seed=42, verbose=False,
            out_type="indices",
        )
        t_v11_time = time.perf_counter() - t0

        # v10
        _set_v10()
        t0 = time.perf_counter()
        t_v10, cl_v10, p_v10, h_v10 = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_permutations,
            threshold=thresh, tail=tail, seed=42, verbose=False,
            out_type="indices",
        )
        t_v10_time = time.perf_counter() - t0
        _set_v11()

        t_match = np.allclose(t_v11, t_v10, atol=1e-10)
        h_match = np.allclose(h_v11, h_v10, atol=1e-10)
        p_match = np.allclose(p_v11, p_v10, atol=1e-10)
        print(f"    v11: {t_v11_time:.3f}s, v10: {t_v10_time:.3f}s, "
              f"speedup: {t_v10_time/t_v11_time:.2f}x")
        print(f"    parity: t_obs={t_match}, H0={h_match}, pvals={p_match}")
        if not (t_match and h_match and p_match):
            if not h_match:
                diff = np.abs(h_v11 - h_v10)
                print(f"    H0 max diff: {diff.max():.2e}")

        del t_v11, cl_v11, p_v11, h_v11, t_v10, cl_v10, p_v10, h_v10
        gc.collect()

    # ==================================================================
    # Scaling test
    # ==================================================================
    print(f"\n{'='*70}")
    print("SCALING (tail=1)")
    print(f"{'='*70}")

    for n_p in [256, 512, 1024, 2048, 4096]:
        # v11
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        t_v11_s = time.perf_counter() - t0
        del _r; gc.collect()

        # v10
        _set_v10()
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        t_v10_s = time.perf_counter() - t0
        _set_v11()
        del _r; gc.collect()

        print(f"  {n_p:5d} perms: v11={t_v11_s:.3f}s  v10={t_v10_s:.3f}s  "
              f"speedup={t_v10_s/t_v11_s:.2f}x  "
              f"({t_v11_s/n_p*1e3:.3f} vs {t_v10_s/n_p*1e3:.3f} ms/perm)")

    # Linear regression for per-perm cost
    times_v11_lr, times_v10_lr, perms_lr = [], [], []
    for n_p in [512, 1024, 2048, 4096]:
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        times_v11_lr.append(time.perf_counter() - t0)
        del _r; gc.collect()

        _set_v10()
        t0 = time.perf_counter()
        _r = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=n_p,
            threshold=threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        times_v10_lr.append(time.perf_counter() - t0)
        _set_v11()
        del _r; gc.collect()

        perms_lr.append(n_p)

    A = np.vstack([perms_lr, np.ones(len(perms_lr))]).T
    slope_v11, intercept_v11 = np.linalg.lstsq(A, times_v11_lr, rcond=None)[0]
    slope_v10, intercept_v10 = np.linalg.lstsq(A, times_v10_lr, rcond=None)[0]
    print(f"\n  v11 linear: overhead={intercept_v11*1e3:.1f}ms, "
          f"per_perm={slope_v11*1e3:.3f}ms")
    print(f"  v10 linear: overhead={intercept_v10*1e3:.1f}ms, "
          f"per_perm={slope_v10*1e3:.3f}ms")

    # ==================================================================
    # Realistic data test (if sample dataset available)
    # ==================================================================
    print(f"\n{'='*70}")
    print("REALISTIC DATA (dSPM source estimates)")
    print(f"{'='*70}")

    try:
        from mne.minimum_norm import apply_inverse, read_inverse_operator

        meg_path = data_path / "MEG" / "sample"
        fname_inv = meg_path / "sample_audvis-meg-oct-6-meg-inv.fif"

        raw = mne.io.read_raw_fif(
            meg_path / "sample_audvis_filt-0-40_raw.fif", verbose=False)
        events = mne.read_events(
            meg_path / "sample_audvis_filt-0-40_raw-eve.fif")
        raw.info["bads"] += ["MEG 2443"]
        picks = mne.pick_types(raw.info, meg=True, eog=True, exclude="bads")
        reject = dict(grad=1000e-13, mag=4000e-15, eog=150e-6)

        epochs1 = mne.Epochs(
            raw, events, 1, -0.2, 0.3, picks=picks,
            baseline=(None, 0), reject=reject, preload=True, verbose=False)
        epochs2 = mne.Epochs(
            raw, events, 3, -0.2, 0.3, picks=picks,
            baseline=(None, 0), reject=reject, preload=True, verbose=False)
        mne.epochs.equalize_epoch_counts([epochs1, epochs2])

        inverse_operator = read_inverse_operator(fname_inv, verbose=False)
        lambda2 = 1.0 / 9.0

        evoked1 = epochs1.average().resample(50, npad="auto", verbose=False)
        evoked2 = epochs2.average().resample(50, npad="auto", verbose=False)
        c1 = apply_inverse(evoked1, inverse_operator, lambda2, "dSPM",
                           verbose=False)
        c2 = apply_inverse(evoked2, inverse_operator, lambda2, "dSPM",
                           verbose=False)
        c1.crop(0, None)
        c2.crop(0, None)

        fsave_vertices = [s["vertno"] for s in src]
        morph_mat = mne.compute_source_morph(
            src=inverse_operator["src"], subject_to="fsaverage",
            spacing=fsave_vertices, subjects_dir=subjects_dir,
            verbose=False).morph_mat

        n_verts_sample, n_t = c1.data.shape
        n_subj_real = 7

        np.random.seed(0)
        X_real = np.random.randn(n_verts_sample, n_t, n_subj_real, 2) * 10
        X_real[:, :, :, 0] += c1.data[:, :, np.newaxis]
        X_real[:, :, :, 1] += c2.data[:, :, np.newaxis]

        n_verts_fsave = morph_mat.shape[0]
        X_real = morph_mat.dot(X_real.reshape(n_verts_sample, -1))
        X_real = X_real.reshape(n_verts_fsave, n_t, n_subj_real, 2)
        X_real = np.abs(X_real)
        X_real = X_real[:, :, :, 0] - X_real[:, :, :, 1]
        X_real = np.transpose(X_real, [2, 1, 0])  # (subjects, time, space)

        real_n_perms = 2048
        real_threshold = 1.67

        print(f"\n  Data: {n_subj_real} subjects × {n_verts_fsave} verts × "
              f"{n_t} times = {n_verts_fsave * n_t:,} tests")

        # v11 real data
        t0 = time.perf_counter()
        t_r11, cl_r11, p_r11, h_r11 = spatio_temporal_cluster_1samp_test(
            X_real, adjacency=adjacency, n_permutations=real_n_perms,
            threshold=real_threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        t_v11_real = time.perf_counter() - t0

        # v10 real data
        _set_v10()
        t0 = time.perf_counter()
        t_r10, cl_r10, p_r10, h_r10 = spatio_temporal_cluster_1samp_test(
            X_real, adjacency=adjacency, n_permutations=real_n_perms,
            threshold=real_threshold, tail=1, seed=42, verbose=False,
            out_type="indices",
        )
        t_v10_real = time.perf_counter() - t0
        _set_v11()

        h_match_real = np.allclose(h_r11, h_r10, atol=1e-10)
        p_match_real = np.allclose(p_r11, p_r10, atol=1e-10)
        print(f"  v11: {t_v11_real:.3f}s  ({t_v11_real/real_n_perms*1e3:.3f} ms/perm)")
        print(f"  v10: {t_v10_real:.3f}s  ({t_v10_real/real_n_perms*1e3:.3f} ms/perm)")
        print(f"  Speedup: {t_v10_real/t_v11_real:.2f}x")
        print(f"  Parity: H0={h_match_real}, pvals={p_match_real}")

        del t_r11, cl_r11, p_r11, h_r11, t_r10, cl_r10, p_r10, h_r10
        gc.collect()

    except Exception as e:
        print(f"  Skipped: {e}")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print("SUMMARY - v11 Parallel Permutation Processing")
    print(f"{'='*70}")
    print(f"Data: {n_subjects} subjects x {n_src} verts x {n_times} times "
          f"= {n_src*n_times:,} tests")
    print(f"Adjacency: fsaverage ico-5 ({n_src} vertices)")
    print()
    print(f"v11 (parallel prange): {v11_median:.3f}s  "
          f"({v11_median/n_permutations*1e3:.3f} ms/perm)")
    print(f"  Linear: overhead={intercept_v11*1e3:.1f}ms, "
          f"per_perm={slope_v11*1e3:.3f}ms")
    print(f"v10 (serial range):    {v10_median:.3f}s  "
          f"({v10_median/n_permutations*1e3:.3f} ms/perm)")
    print(f"  Linear: overhead={intercept_v10*1e3:.1f}ms, "
          f"per_perm={slope_v10*1e3:.3f}ms")
    print()
    print(f"End-to-end speedup ({n_permutations} perms): "
          f"{v10_median/v11_median:.2f}x")
    print(f"Per-perm speedup (linear slope): "
          f"{slope_v10/slope_v11:.2f}x")
    print(f"Parity: {'ALL PASS' if all_match else 'FAIL'}")
    print()
    print(f"Projected at 5000 perms:")
    t_v11_5k = intercept_v11 + 5000 * slope_v11
    t_v10_5k = intercept_v10 + 5000 * slope_v10
    print(f"  v11: {t_v11_5k:.1f}s")
    print(f"  v10: {t_v10_5k:.1f}s")
    print(f"  Savings: {t_v10_5k - t_v11_5k:.1f}s "
          f"({(1-t_v11_5k/t_v10_5k)*100:.0f}% faster)")


if __name__ == "__main__":
    run_benchmark()
