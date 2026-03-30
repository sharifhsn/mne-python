#!/usr/bin/env python3
"""
A/B benchmark: perf-opt vs baseline MNE permutation cluster test.

Uses 15 synthetic subjects on fsaverage ico-5 (20,484 vertices × 15 times
= 307,260 tests) to ensure enough permutation headroom for large sweeps.
Extracts overhead + per-perm cost via linear regression.
"""

import sys
import time

import numpy as np
from scipy import stats as sp_stats

import mne
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.stats import cluster_level as cl


def load_source_data():
    """Set up fsaverage ico-5 adjacency + synthetic random data."""
    subjects_dir = mne.datasets.sample.data_path() / "subjects"
    src = mne.setup_source_space(
        "fsaverage", spacing="ico5", subjects_dir=subjects_dir, add_dist=False
    )
    adjacency = mne.spatial_src_adjacency(src)
    n_src = adjacency.shape[0]

    n_subjects, n_times = 15, 15
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_subjects, n_times, n_src))
    # Inject signal in a small region so clusters form
    sig_verts = np.arange(100, 200)
    sig_times = np.arange(5, 10)
    X[:, sig_times[:, None], sig_verts[None, :]] += 2.0

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
    print("PERF-OPT vs BASELINE BENCHMARK")
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

    threshold = 1.67  # same as v13 benchmarks
    max_perms = 2 ** (n_subjects) - 1  # tail=1 so no halving
    print(f"threshold: {threshold}")
    print(f"max_perms (15 subjects, tail=1): {max_perms}")

    perm_counts = [256, 512, 1024, 2048, 4096]

    # --- Warmup run (JIT compilation) ---
    print("\n--- Warmup (64 perms) ---")
    run_sweep(X, adjacency, threshold, [64], seed=42, tag="warmup")

    # --- Optimized run (perf-opt) ---
    print("\n--- OPTIMIZED (perf-opt) ---")
    opt_results = run_sweep(X, adjacency, threshold, perm_counts,
                            seed=42, tag="OPT")
    opt_overhead, opt_slope, opt_r2 = linear_regression(opt_results)

    # --- Baseline run (disable Numba to use pure Python/SciPy fallbacks) ---
    print("\n--- BASELINE (has_numba=False, disables compact graph + UF) ---")
    saved_has_numba = cl.has_numba
    cl.has_numba = False
    # Warmup baseline too
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

    # Projected times
    print(f"\n  Projected 10,000 perms:")
    print(f"    Optimized: {opt_overhead/1000 + 10000*opt_slope/1000:.1f}s")
    print(f"    Baseline:  {base_overhead/1000 + 10000*base_slope/1000:.1f}s")


if __name__ == "__main__":
    main()
