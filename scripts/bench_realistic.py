#!/usr/bin/env python3
"""
Calibrated benchmark: perf-opt vs baseline with realistic cluster density.

Previous benchmark used threshold=1.67 on random data, producing ~50%
supra-threshold density (13,974 clusters). Real neuroimaging source
estimates (dSPM/MNE) typically have ~1-5% supra-threshold density with
50-500 clusters.

This benchmark uses threshold=3.0 (matching typical group-level t-stat
threshold) and injects focal activations in regions matching the face
processing network (fusiform, occipital). This produces realistic
cluster density for honest speedup measurements.
"""

import sys
import time

import numpy as np
from scipy import stats as sp_stats

import mne
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.stats import cluster_level as cl


def load_source_data():
    """Set up fsaverage ico-5 adjacency + realistic synthetic data."""
    subjects_dir = mne.datasets.sample.data_path() / "subjects"
    src = mne.setup_source_space(
        "fsaverage", spacing="ico5", subjects_dir=subjects_dir, add_dist=False
    )
    adjacency = mne.spatial_src_adjacency(src)
    n_src = adjacency.shape[0]

    n_subjects, n_times = 15, 15
    rng = np.random.default_rng(42)

    # Unit-variance Gaussian noise (so t-stats ~ t-distribution under null)
    X = rng.standard_normal((n_subjects, n_times, n_src))

    # Inject focal activations mimicking face processing network.
    # Effect size ~1.0 std (moderate, realistic for group-level MEG).
    # Left fusiform region
    X[:, 5:10, 1000:1100] += 1.0
    # Right fusiform region
    X[:, 5:10, 11000:11080] += 1.0
    # Left occipital (early visual)
    X[:, 3:8, 3000:3060] += 0.8

    return X, adjacency, n_subjects


def run_sweep(X, adjacency, threshold, perm_counts, seed=42, tag=""):
    """Run benchmark across multiple perm counts, return timings."""
    results = []
    for n_perms in perm_counts:
        t0 = time.perf_counter()
        T_obs, clusters, pvals, H0 = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_jobs=1,
            threshold=threshold, n_permutations=n_perms,
            buffer_size=None, verbose=False, seed=seed, tail=1,
        )
        elapsed = time.perf_counter() - t0
        results.append({
            "n_perms": n_perms,
            "elapsed": elapsed,
            "n_clusters": len(clusters),
            "min_p": pvals.min() if len(clusters) > 0 else 1.0,
            "H0_hash": hash(H0.tobytes()),
        })
        print(f"  {tag} n_perms={n_perms:5d}: {elapsed:7.3f}s "
              f"({len(clusters)} clusters, min_p={results[-1]['min_p']:.4f})")
    return results


def linear_regression(results):
    """Extract overhead + per-perm cost from sweep results."""
    x = np.array([r["n_perms"] for r in results], dtype=float)
    y = np.array([r["elapsed"] for r in results], dtype=float)
    slope, intercept, r_value, _, _ = sp_stats.linregress(x, y)
    return intercept * 1000, slope * 1000, r_value**2  # ms


def main():
    import os

    print("=" * 70)
    print("CALIBRATED REALISTIC BENCHMARK")
    print("=" * 70)

    print(f"\nCPU cores: {os.cpu_count()}")
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    print(f"CPU model: {line.split(':')[1].strip()}")
                    break
    except FileNotFoundError:
        pass

    print(f"Python: {sys.version.split()[0]}")
    print(f"MNE: {mne.__version__}")
    print(f"NumPy: {np.__version__}")

    has_numba = cl.has_numba
    print(f"Numba available: {has_numba}")

    print("\nLoading source-space data...")
    X, adjacency, n_subjects = load_source_data()
    print(f"Data shape: {X.shape}")
    print(f"Adjacency: {adjacency.shape}, nnz={adjacency.nnz}")
    n_tests = X.shape[1] * X.shape[2]
    print(f"n_tests: {n_tests:,}")

    threshold = 3.0  # realistic group-level threshold
    max_perms = 2 ** n_subjects - 1
    print(f"threshold: {threshold}")
    print(f"max_perms (15 subjects, tail=1): {max_perms}")

    # Density diagnostic: check how many vertices exceed threshold
    # in the observed t-stat map
    t_obs_diag = np.mean(X, axis=0) / (np.std(X, axis=0, ddof=1)
                                        / np.sqrt(n_subjects))
    n_suprathresh = np.sum(np.abs(t_obs_diag) > threshold)
    pct = 100.0 * n_suprathresh / t_obs_diag.size
    print(f"\nDensity diagnostic:")
    print(f"  Supra-threshold tests: {n_suprathresh:,} / {t_obs_diag.size:,} "
          f"({pct:.2f}%)")

    perm_counts = [256, 512, 1024, 2048, 4096]

    # --- Warmup (JIT compilation) ---
    print("\n--- Warmup (64 perms) ---")
    run_sweep(X, adjacency, threshold, [64], seed=42, tag="warmup")

    # --- Optimized run ---
    print("\n--- OPTIMIZED (perf-opt) ---")
    opt_results = run_sweep(X, adjacency, threshold, perm_counts,
                            seed=42, tag="OPT")
    opt_overhead, opt_slope, opt_r2 = linear_regression(opt_results)

    # --- Baseline run ---
    print("\n--- BASELINE (has_numba=False) ---")
    saved_has_numba = cl.has_numba
    cl.has_numba = False
    run_sweep(X, adjacency, threshold, [64], seed=42, tag="base-warmup")
    base_results = run_sweep(X, adjacency, threshold, perm_counts,
                             seed=42, tag="BASE")
    cl.has_numba = saved_has_numba
    base_overhead, base_slope, base_r2 = linear_regression(base_results)

    # --- Parity check ---
    print("\n--- PARITY CHECK ---")
    all_pass = True
    for opt, base in zip(opt_results, base_results):
        h0_match = opt["H0_hash"] == base["H0_hash"]
        status = "PASS" if h0_match else "FAIL"
        if not h0_match:
            all_pass = False
        print(f"  n_perms={opt['n_perms']:5d}: H0 {status}")
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("LINEAR REGRESSION")
    print("=" * 70)
    print(f"  {'Version':<12} {'Overhead (ms)':>14} {'Slope (ms/perm)':>16} {'R²':>6}")
    print(f"  {'Optimized':<12} {opt_overhead:>14.1f} {opt_slope:>16.3f} {opt_r2:>6.4f}")
    print(f"  {'Baseline':<12} {base_overhead:>14.1f} {base_slope:>16.3f} {base_r2:>6.4f}")

    if opt_overhead > 0:
        print(f"\n  Overhead speedup: {base_overhead/opt_overhead:.1f}x")
    print(f"  Per-perm speedup: {base_slope/opt_slope:.1f}x")

    print("\n" + "=" * 70)
    print("END-TO-END COMPARISON")
    print("=" * 70)
    print(f"  {'n_perms':>8} {'Optimized':>10} {'Baseline':>10} {'Speedup':>8}")
    for opt, base in zip(opt_results, base_results):
        speedup = base["elapsed"] / opt["elapsed"]
        print(f"  {opt['n_perms']:>8d} {opt['elapsed']:>9.3f}s "
              f"{base['elapsed']:>9.3f}s {speedup:>7.1f}x")

    print(f"\n  Projected 10,000 perms:")
    print(f"    Optimized: {opt_overhead/1000 + 10000*opt_slope/1000:.1f}s")
    print(f"    Baseline:  {base_overhead/1000 + 10000*base_slope/1000:.1f}s")


if __name__ == "__main__":
    main()
