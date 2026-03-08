#!/usr/bin/env python3
"""
Benchmark: MNE permutation cluster test -- CPU baseline.

This script establishes CPU baseline timing for the spatio-temporal
cluster-based permutation test on source-space data (fsaverage ico-5).

This is the main bottleneck researchers wait on. GPU acceleration targets
the connected-component labeling step which consumes ~97% of the time.

Setup (Linux workstation):
    uv venv --python 3.12 .venv
    uv pip install -e ".[full-no-qt]" --group test
    # Download sample data (~1.5GB, one-time):
    python -c "import mne; mne.datasets.sample.data_path()"

Usage:
    python gpu_accel/benchmark_cluster_cpu.py
"""

import sys
import time

import numpy as np
from scipy import stats

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.stats import spatio_temporal_cluster_1samp_test


def load_source_data():
    """Load and prepare source-space data from MNE sample dataset."""
    data_path = sample.data_path()
    meg_path = data_path / "MEG" / "sample"
    subjects_dir = data_path / "subjects"
    src_fname = subjects_dir / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif"
    fname_inv = meg_path / "sample_audvis-meg-oct-6-meg-inv.fif"

    raw = mne.io.read_raw_fif(
        meg_path / "sample_audvis_filt-0-40_raw.fif", verbose=False
    )
    events = mne.read_events(meg_path / "sample_audvis_filt-0-40_raw-eve.fif")
    raw.info["bads"] += ["MEG 2443"]
    picks = mne.pick_types(raw.info, meg=True, eog=True, exclude="bads")
    reject = dict(grad=1000e-13, mag=4000e-15, eog=150e-6)

    epochs1 = mne.Epochs(
        raw, events, 1, -0.2, 0.3, picks=picks,
        baseline=(None, 0), reject=reject, preload=True, verbose=False,
    )
    epochs2 = mne.Epochs(
        raw, events, 3, -0.2, 0.3, picks=picks,
        baseline=(None, 0), reject=reject, preload=True, verbose=False,
    )
    mne.epochs.equalize_epoch_counts([epochs1, epochs2])

    inverse_operator = read_inverse_operator(fname_inv, verbose=False)
    snr, method = 3.0, "dSPM"
    lambda2 = 1.0 / snr**2

    evoked1 = epochs1.average().resample(50, npad="auto", verbose=False)
    evoked2 = epochs2.average().resample(50, npad="auto", verbose=False)
    condition1 = apply_inverse(evoked1, inverse_operator, lambda2, method,
                               verbose=False)
    condition2 = apply_inverse(evoked2, inverse_operator, lambda2, method,
                               verbose=False)
    condition1.crop(0, None)
    condition2.crop(0, None)

    src = mne.read_source_spaces(src_fname, verbose=False)
    fsave_vertices = [s["vertno"] for s in src]
    morph_mat = mne.compute_source_morph(
        src=inverse_operator["src"], subject_to="fsaverage",
        spacing=fsave_vertices, subjects_dir=subjects_dir, verbose=False,
    ).morph_mat

    n_vertices_sample, n_times = condition1.data.shape
    n_subjects = 7

    np.random.seed(0)
    X = np.random.randn(n_vertices_sample, n_times, n_subjects, 2) * 10
    X[:, :, :, 0] += condition1.data[:, :, np.newaxis]
    X[:, :, :, 1] += condition2.data[:, :, np.newaxis]

    n_vertices_fsave = morph_mat.shape[0]
    X = morph_mat.dot(X.reshape(n_vertices_sample, -1))
    X = X.reshape(n_vertices_fsave, n_times, n_subjects, 2)
    X = np.abs(X)
    X = X[:, :, :, 0] - X[:, :, :, 1]
    X = np.transpose(X, [2, 1, 0])  # (subjects, time, space)

    adjacency = mne.spatial_src_adjacency(src)

    return X, adjacency, n_subjects


def benchmark(X, adjacency, n_subjects, n_perms_list=None, n_jobs_list=None):
    """Run benchmark across different parameter combinations."""
    if n_perms_list is None:
        n_perms_list = [64, 256, 1024]
    if n_jobs_list is None:
        n_jobs_list = [1]

    df = n_subjects - 1
    t_threshold = stats.distributions.t.ppf(1 - 0.001 / 2, df=df)

    print(f"\nData shape: {X.shape}")
    print(f"Adjacency: {adjacency.shape}, nnz={adjacency.nnz}")
    print(f"n_tests: {X.shape[1] * X.shape[2]:,}")
    print(f"t_threshold: {t_threshold:.3f}")
    print("-" * 70)

    results = []
    for n_perms in n_perms_list:
        for n_jobs in n_jobs_list:
            t0 = time.perf_counter()
            T_obs, clusters, cluster_p_values, H0 = \
                spatio_temporal_cluster_1samp_test(
                    X, adjacency=adjacency, n_jobs=n_jobs,
                    threshold=t_threshold, n_permutations=n_perms,
                    buffer_size=None, verbose=False, seed=42,
                )
            elapsed = time.perf_counter() - t0
            result = {
                "n_perms": n_perms,
                "n_jobs": n_jobs,
                "elapsed": elapsed,
                "n_clusters": len(clusters),
                "min_p": cluster_p_values.min() if len(clusters) > 0 else 1.0,
            }
            results.append(result)
            print(
                f"n_perms={n_perms:5d}, n_jobs={n_jobs}: "
                f"{elapsed:7.1f}s  ({len(clusters)} clusters, "
                f"min p={result['min_p']:.4f})"
            )

    return results


def main():
    import os

    # Detect available CPU cores
    n_cores = os.cpu_count() or 1
    print(f"CPU cores detected: {n_cores}")

    # Choose n_jobs based on cores
    n_jobs_list = [1]
    if n_cores >= 4:
        n_jobs_list.append(min(n_cores, 6))
    if n_cores >= 8:
        n_jobs_list.append(n_cores)

    print("Loading source-space data from MNE sample dataset...")
    X, adjacency, n_subjects = load_source_data()
    print("Data loaded.\n")

    print("=" * 70)
    print("CPU BASELINE BENCHMARK")
    print("=" * 70)
    results = benchmark(
        X, adjacency, n_subjects,
        n_perms_list=[64, 256, 1024],
        n_jobs_list=n_jobs_list,
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        rate = r["n_perms"] / r["elapsed"]
        print(
            f"  {r['n_perms']:5d} perms, {r['n_jobs']:2d} jobs: "
            f"{r['elapsed']:6.1f}s ({rate:.1f} perms/s)"
        )

    # Estimate time for common real-world workloads
    if results:
        # Use single-core 1024-perm rate for conservative estimate
        ref = [r for r in results if r["n_perms"] == 1024 and r["n_jobs"] == 1]
        if ref:
            rate_1core = ref[0]["n_perms"] / ref[0]["elapsed"]
            print(f"\n  Projected time for 10,000 permutations (1 core): "
                  f"{10000 / rate_1core:.0f}s "
                  f"({10000 / rate_1core / 60:.1f} min)")


if __name__ == "__main__":
    main()
