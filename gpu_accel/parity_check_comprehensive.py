#!/usr/bin/env python
"""Comprehensive parity check: current branch vs baseline.

Runs many different workflow configurations of the cluster permutation tests
and saves all results (t_obs, clusters, p-values, H0) to a JSON-compatible
numpy file for comparison between branches.

Usage:
    python gpu_accel/parity_check_comprehensive.py [output_prefix]

Output: {output_prefix}_results.npz with all test results.
"""

import sys
import time
import traceback

import numpy as np

import mne
from mne.stats import (
    permutation_cluster_1samp_test,
    permutation_cluster_test,
    spatio_temporal_cluster_1samp_test,
    spatio_temporal_cluster_test,
)
from mne.stats.cluster_level import ttest_1samp_no_p


def custom_stat_fun(X):
    """Custom stat function (same as ttest but not identity-equal)."""
    return ttest_1samp_no_p(X)


def run_all_tests():
    """Run comprehensive parity tests and return results dict."""
    results = {}
    test_num = 0

    # ── Setup data ──────────────────────────────────────────────────────
    print("Setting up data...")
    subjects_dir = mne.datasets.sample.data_path() / "subjects"
    src = mne.setup_source_space(
        "fsaverage", spacing="ico5", subjects_dir=subjects_dir, add_dist=False
    )
    adjacency = mne.spatial_src_adjacency(src)
    n_src = adjacency.shape[0]

    # Standard spatio-temporal data
    n_subjects, n_times = 15, 15
    rng = np.random.default_rng(42)
    X_st = rng.standard_normal((n_subjects, n_times, n_src))
    # Add signal
    sig_verts = np.arange(100, 200)
    sig_times = np.arange(5, 10)
    X_st[:, sig_times[:, None], sig_verts[None, :]] += 2.0

    # 1D data (no time dimension, just vertices)
    X_1d = rng.standard_normal((n_subjects, n_src))
    X_1d[:, 100:200] += 1.5

    # Small data for exact test (n_samples=6, max_perms=63)
    X_small = rng.standard_normal((6, 10, 50))
    X_small[:, 3:7, 10:20] += 3.0

    # Two-sample data for F-test
    X_group1 = rng.standard_normal((10, n_times, 200))
    X_group2 = rng.standard_normal((10, n_times, 200))
    X_group1[:, 3:7, 50:80] += 2.0

    # Small adjacency for two-sample tests
    from scipy import sparse
    n_small = 200
    diags = [np.ones(n_small - 1), np.ones(n_small - 1)]
    adj_small = sparse.diags(diags, [-1, 1], shape=(n_small, n_small)).tocsr()

    # Data with no signal (edge case: no clusters)
    X_nosig = rng.standard_normal((n_subjects, n_times, n_src)) * 0.5

    # Data with very strong signal (edge case: almost all significant)
    X_allsig = rng.standard_normal((n_subjects, n_times, 200))
    X_allsig += 5.0

    print(f"Source space: {n_src} vertices")
    print(f"Spatio-temporal shape: {X_st.shape}")
    print()

    def save_result(name, t_obs, clusters, pv, H0):
        """Save a test result."""
        nonlocal test_num
        test_num += 1
        results[f"{test_num:03d}_{name}_t_obs"] = np.array(t_obs).ravel()
        results[f"{test_num:03d}_{name}_H0"] = np.array(H0).ravel()
        results[f"{test_num:03d}_{name}_pv"] = np.array(pv).ravel()
        results[f"{test_num:03d}_{name}_n_clusters"] = np.array([len(clusters)])
        # Save cluster sizes (not the full indices, which depend on format)
        if len(clusters) > 0:
            if isinstance(clusters[0], np.ndarray):
                if clusters[0].dtype == bool:
                    sizes = np.array([np.sum(c) for c in clusters])
                else:
                    sizes = np.array([len(c) if c.ndim == 1 else np.prod([len(idx) for idx in c]) for c in clusters])
            elif isinstance(clusters[0], tuple):
                sizes = np.array([len(c[0]) if isinstance(c[0], np.ndarray) else 1 for c in clusters])
            elif isinstance(clusters[0], slice):
                sizes = np.array([c.stop - c.start for c in clusters])
            else:
                sizes = np.array([0])
        else:
            sizes = np.array([])
        results[f"{test_num:03d}_{name}_cluster_sizes"] = sizes
        print(f"  [{test_num:03d}] {name}: {len(clusters)} clusters, "
              f"H0 range=[{np.min(H0):.4f}, {np.max(H0):.4f}]")

    # ── Test Group 1: Spatio-temporal, different tails ──────────────────
    print("=== Group 1: Spatio-temporal, different tails ===")

    for tail in [1, 0, -1]:
        thresh = 1.67 if tail >= 0 else -1.67
        t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
            X_st, threshold=thresh, adjacency=adjacency,
            n_permutations=256, tail=tail, seed=42,
            out_type="indices", verbose=False,
        )
        save_result(f"st_tail{tail:+d}_idx", t_obs, clusters, pv, H0)

    # ── Test Group 2: Different out_types ───────────────────────────────
    print("\n=== Group 2: Different out_types ===")

    t_obs, clusters_mask, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_st, threshold=1.67, adjacency=adjacency,
        n_permutations=256, tail=1, seed=42,
        out_type="mask", verbose=False,
    )
    save_result("st_mask", t_obs, clusters_mask, pv, H0)

    t_obs, clusters_idx, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_st, threshold=1.67, adjacency=adjacency,
        n_permutations=256, tail=1, seed=42,
        out_type="indices", verbose=False,
    )
    save_result("st_indices", t_obs, clusters_idx, pv, H0)

    # ── Test Group 3: Different thresholds ──────────────────────────────
    print("\n=== Group 3: Different thresholds ===")

    for thresh in [1.0, 1.67, 2.5, 3.5]:
        t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
            X_st, threshold=thresh, adjacency=adjacency,
            n_permutations=256, tail=1, seed=42,
            out_type="indices", verbose=False,
        )
        save_result(f"st_thresh{thresh}", t_obs, clusters, pv, H0)

    # ── Test Group 4: Auto threshold (None) ─────────────────────────────
    print("\n=== Group 4: Auto threshold ===")

    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_st, threshold=None, adjacency=adjacency,
        n_permutations=256, tail=1, seed=42,
        out_type="indices", verbose=False,
    )
    save_result("st_auto_thresh_tail1", t_obs, clusters, pv, H0)

    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_st, threshold=None, adjacency=adjacency,
        n_permutations=256, tail=0, seed=42,
        out_type="indices", verbose=False,
    )
    save_result("st_auto_thresh_tail0", t_obs, clusters, pv, H0)

    # ── Test Group 5: Different seeds ───────────────────────────────────
    print("\n=== Group 5: Different seeds ===")

    for seed in [0, 42, 123, 9999]:
        t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
            X_st, threshold=1.67, adjacency=adjacency,
            n_permutations=256, tail=1, seed=seed,
            out_type="indices", verbose=False,
        )
        save_result(f"st_seed{seed}", t_obs, clusters, pv, H0)

    # ── Test Group 6: Different perm counts ─────────────────────────────
    print("\n=== Group 6: Different perm counts ===")

    for n_perms in [64, 128, 256, 512, 1024]:
        t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
            X_st, threshold=1.67, adjacency=adjacency,
            n_permutations=n_perms, tail=1, seed=42,
            out_type="indices", verbose=False,
        )
        save_result(f"st_nperms{n_perms}", t_obs, clusters, pv, H0)

    # ── Test Group 7: Custom stat function (non-identity) ───────────────
    print("\n=== Group 7: Custom stat function ===")

    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_st, threshold=1.67, adjacency=adjacency,
        n_permutations=128, tail=1, seed=42, stat_fun=custom_stat_fun,
        out_type="indices", verbose=False,
    )
    save_result("st_custom_stat", t_obs, clusters, pv, H0)

    # ── Test Group 8: 1D data (no time dimension) ──────────────────────
    print("\n=== Group 8: 1D data with adjacency ===")

    # Use spatial adjacency directly (n_tests == adjacency.shape[0])
    t_obs, clusters, pv, H0 = permutation_cluster_1samp_test(
        X_1d, threshold=1.67, adjacency=adjacency,
        n_permutations=256, tail=1, seed=42,
        out_type="indices", verbose=False,
    )
    save_result("1d_sparse_adj", t_obs, clusters, pv, H0)

    # ── Test Group 9: No adjacency (ndimage path) ──────────────────────
    print("\n=== Group 9: No adjacency (ndimage) ===")

    X_2d_data = rng.standard_normal((15, 10, 20))
    X_2d_data[:, 3:7, 5:15] += 2.5

    t_obs, clusters, pv, H0 = permutation_cluster_1samp_test(
        X_2d_data, threshold=1.67, adjacency=None,
        n_permutations=256, tail=1, seed=42,
        out_type="mask", verbose=False,
    )
    save_result("no_adj_mask", t_obs, clusters, pv, H0)

    t_obs, clusters, pv, H0 = permutation_cluster_1samp_test(
        X_2d_data, threshold=1.67, adjacency=None,
        n_permutations=256, tail=0, seed=42,
        out_type="indices", verbose=False,
    )
    save_result("no_adj_tail0_idx", t_obs, clusters, pv, H0)

    # ── Test Group 10: Exact test (small n_samples) ─────────────────────
    print("\n=== Group 10: Exact test (small n_samples) ===")

    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_small, threshold=1.67, adjacency=adj_small[:50, :50],
        n_permutations="all", tail=1, seed=42,
        out_type="indices", verbose=False,
    )
    save_result("exact_test_tail1", t_obs, clusters, pv, H0)

    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_small, threshold=1.67, adjacency=adj_small[:50, :50],
        n_permutations="all", tail=0, seed=42,
        out_type="indices", verbose=False,
    )
    save_result("exact_test_tail0", t_obs, clusters, pv, H0)

    # ── Test Group 11: No signal (edge case) ────────────────────────────
    print("\n=== Group 11: No signal (edge case) ===")

    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_nosig, threshold=3.0, adjacency=adjacency,
        n_permutations=128, tail=1, seed=42,
        out_type="indices", verbose=False,
    )
    save_result("no_signal", t_obs, clusters, pv, H0)

    # ── Test Group 12: Strong signal (nearly all significant) ───────────
    print("\n=== Group 12: Strong signal ===")

    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_allsig, threshold=1.67, adjacency=adj_small,
        n_permutations=128, tail=1, seed=42,
        out_type="indices", verbose=False,
    )
    save_result("strong_signal", t_obs, clusters, pv, H0)

    # ── Test Group 13: Two-sample F-test ────────────────────────────────
    print("\n=== Group 13: Two-sample F-test ===")

    t_obs, clusters, pv, H0 = permutation_cluster_test(
        [X_group1.reshape(10, -1), X_group2.reshape(10, -1)],
        threshold=3.0, adjacency=None,
        n_permutations=128, tail=1, seed=42,
        out_type="mask", verbose=False,
    )
    save_result("fstest_no_adj", t_obs, clusters, pv, H0)

    # F-test with adjacency
    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_group1 - X_group2,  # Paired difference for 1-samp test
        threshold=1.67, adjacency=adj_small,
        n_permutations=128, tail=0, seed=42,
        out_type="indices", verbose=False,
    )
    save_result("paired_diff_adj", t_obs, clusters, pv, H0)

    # ── Test Group 14: With exclude mask ────────────────────────────────
    print("\n=== Group 14: With exclude mask ===")

    # exclude requires adjacency.shape[0] == n_tests (global adjacency path)
    exclude_1d = np.zeros(n_src, dtype=bool)
    exclude_1d[:5000] = True  # Exclude first 5000 vertices

    t_obs, clusters, pv, H0 = permutation_cluster_1samp_test(
        X_1d, threshold=1.67, adjacency=adjacency,
        n_permutations=128, tail=1, seed=42, exclude=exclude_1d,
        out_type="indices", verbose=False,
    )
    save_result("with_exclude", t_obs, clusters, pv, H0)

    # ── Test Group 15: With buffer_size ─────────────────────────────────
    print("\n=== Group 15: With buffer_size ===")

    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_st, threshold=1.67, adjacency=adjacency,
        n_permutations=128, tail=1, seed=42, buffer_size=10000,
        out_type="indices", verbose=False,
    )
    save_result("buffer_10000", t_obs, clusters, pv, H0)

    # ── Test Group 16: step_down_p > 0 ──────────────────────────────────
    print("\n=== Group 16: step_down_p > 0 ===")

    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_st, threshold=1.67, adjacency=adjacency,
        n_permutations=128, tail=1, seed=42, step_down_p=0.05,
        out_type="indices", verbose=False,
    )
    save_result("step_down_p", t_obs, clusters, pv, H0)

    # ── Test Group 17: TFCE ─────────────────────────────────────────────
    print("\n=== Group 17: TFCE ===")

    X_tfce = rng.standard_normal((15, 10, 20))
    X_tfce[:, 3:7, 5:15] += 2.0
    tfce_threshold = dict(start=0, step=0.2)

    t_obs, clusters, pv, H0 = permutation_cluster_1samp_test(
        X_tfce, threshold=tfce_threshold, adjacency=None,
        n_permutations=64, tail=1, seed=42,
        out_type="mask", verbose=False,
    )
    save_result("tfce_tail1", t_obs, clusters, pv, H0)

    t_obs, clusters, pv, H0 = permutation_cluster_1samp_test(
        X_tfce, threshold=tfce_threshold, adjacency=None,
        n_permutations=64, tail=0, seed=42,
        out_type="mask", verbose=False,
    )
    save_result("tfce_tail0", t_obs, clusters, pv, H0)

    # ── Test Group 18: adjacency=False ──────────────────────────────────
    print("\n=== Group 18: adjacency=False ===")

    X_false = rng.standard_normal((15, 50))
    X_false[:, 10:30] += 2.0

    t_obs, clusters, pv, H0 = permutation_cluster_1samp_test(
        X_false, threshold=1.67, adjacency=False,
        n_permutations=128, tail=1, seed=42,
        out_type="mask", verbose=False,
    )
    save_result("adj_false", t_obs, clusters, pv, H0)

    # ── Test Group 19: n_jobs > 1 ──────────────────────────────────────
    print("\n=== Group 19: n_jobs > 1 ===")

    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_st, threshold=1.67, adjacency=adjacency,
        n_permutations=256, tail=1, seed=42, n_jobs=2,
        out_type="indices", verbose=False,
    )
    save_result("njobs2", t_obs, clusters, pv, H0)

    # ── Test Group 20: Two-sample with adjacency ────────────────────────
    print("\n=== Group 20: Two-sample with adjacency ===")

    t_obs, clusters, pv, H0 = spatio_temporal_cluster_test(
        [X_group1, X_group2],
        threshold=3.0, adjacency=adj_small,
        n_permutations=128, tail=1, seed=42,
        out_type="indices", verbose=False,
    )
    save_result("twosamp_adj", t_obs, clusters, pv, H0)

    # ── Test Group 21: Large perm count ─────────────────────────────────
    print("\n=== Group 21: Large perm count (2048) ===")

    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_st, threshold=1.67, adjacency=adjacency,
        n_permutations=2048, tail=1, seed=42,
        out_type="indices", verbose=False,
    )
    save_result("st_2048perms", t_obs, clusters, pv, H0)

    # ── Test Group 22: t_power != 1 ────────────────────────────────────
    print("\n=== Group 22: t_power != 1 ===")

    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_st, threshold=1.67, adjacency=adjacency,
        n_permutations=128, tail=1, seed=42, t_power=2,
        out_type="indices", verbose=False,
    )
    save_result("t_power2", t_obs, clusters, pv, H0)

    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_st, threshold=1.67, adjacency=adjacency,
        n_permutations=128, tail=1, seed=42, t_power=0,
        out_type="indices", verbose=False,
    )
    save_result("t_power0", t_obs, clusters, pv, H0)

    # ── Test Group 23: Negative tail with positive data ────────────────
    print("\n=== Group 23: Negative tail edge cases ===")

    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_st, threshold=-1.67, adjacency=adjacency,
        n_permutations=128, tail=-1, seed=42,
        out_type="indices", verbose=False,
    )
    save_result("neg_tail_pos_data", t_obs, clusters, pv, H0)

    # Negative data
    X_neg = -X_st.copy()
    t_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X_neg, threshold=-1.67, adjacency=adjacency,
        n_permutations=128, tail=-1, seed=42,
        out_type="indices", verbose=False,
    )
    save_result("neg_tail_neg_data", t_obs, clusters, pv, H0)

    print(f"\n=== Total: {test_num} test configurations ===")
    return results


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "parity"
    print(f"Output prefix: {prefix}")
    print()

    t0 = time.perf_counter()
    results = run_all_tests()
    elapsed = time.perf_counter() - t0

    output_path = f"/tmp/{prefix}_results.npz"
    np.savez_compressed(output_path, **results)
    print(f"\nSaved {len(results)} arrays to {output_path}")
    print(f"Total time: {elapsed:.1f}s")
