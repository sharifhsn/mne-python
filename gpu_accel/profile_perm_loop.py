#!/usr/bin/env python3
"""
Profile the per-permutation hot path to find the next optimization target.

Instruments:
  1. ttest computation (signs @ X, variance, t-stat)
  2. thresholding (x > thresh, np.where)
  3. _fused_ccl (Numba union-find)
  4. bincount sums
  5. overhead (argmax, bookkeeping)
"""

import time

import numpy as np
from scipy import sparse, stats

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.stats.cluster_level import (
    _find_clusters,
    _fused_ccl,
    _get_components,
    ttest_1samp_no_p,
)


def load_source_data():
    """Load fsaverage source-space data (same as benchmark_cluster_cpu.py)."""
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
    condition1 = apply_inverse(evoked1, inverse_operator, lambda2, method, verbose=False)
    condition2 = apply_inverse(evoked2, inverse_operator, lambda2, method, verbose=False)
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


def profile_perm_loop(X, adjacency, n_subjects, n_perms=256):
    """Profile the inner permutation loop step by step."""
    n_samp, n_times, n_vertices = X.shape
    # Reshape to 2D as the actual function does
    X_2d = X.reshape(n_samp, -1)
    n_tests = X_2d.shape[1]

    df = n_subjects - 1
    threshold = stats.distributions.t.ppf(1 - 0.001 / 2, df=df)

    # Ensure adjacency is COO
    if sparse.issparse(adjacency):
        adjacency_coo = sparse.coo_array(adjacency)
    else:
        adjacency_coo = adjacency

    # Build spatio-temporal adjacency (same as _permutation_cluster_test does)
    from mne.stats.cluster_level import _setup_adjacency
    adjacency_full = _setup_adjacency(adjacency_coo, n_times, n_vertices)
    adj_row = adjacency_full.row.astype(np.intp)
    adj_col = adjacency_full.col.astype(np.intp)

    # Precompute constants (same as v5/v6)
    _sum_sq = np.sum(X_2d ** 2, axis=0)
    _sqrt_n_nm1 = np.sqrt(n_samp * (n_samp - 1))
    _mean_s = np.empty(n_tests, dtype=np.float64)
    _denom_sq = np.empty(n_tests, dtype=np.float64)
    _t_buf = np.empty(n_tests, dtype=np.float64)

    # Generate random orders
    rng = np.random.RandomState(42)
    orders = [rng.choice([False, True], size=n_samp) for _ in range(n_perms)]

    # Warmup: run _fused_ccl once to trigger JIT compilation
    print("Warming up Numba JIT...")
    dummy_x_in = np.ones(n_tests, dtype=bool)
    dummy_idx = np.arange(n_tests, dtype=np.intp)
    _ = _fused_ccl(dummy_x_in, adj_row, adj_col, len(dummy_idx), dummy_idx)
    print("JIT warm.")
    print()

    # Accumulate times
    t_ttest = []
    t_threshold = []
    t_where = []
    t_ccl = []
    t_bincount = []
    t_overhead = []
    t_total = []

    for pi, order in enumerate(orders):
        t0_total = time.perf_counter()

        # 1. Precomputed ttest
        t0 = time.perf_counter()
        signs_1d = 2.0 * order - 1.0
        np.dot(signs_1d, X_2d, out=_mean_s)
        _mean_s /= n_samp
        np.multiply(_mean_s, _mean_s, out=_denom_sq)
        _denom_sq *= -n_samp
        _denom_sq += _sum_sq
        np.maximum(_denom_sq, 0, out=_denom_sq)
        np.sqrt(_denom_sq, out=_denom_sq)
        np.divide(_mean_s, _denom_sq, out=_t_buf)
        _t_buf *= _sqrt_n_nm1
        t1 = time.perf_counter()
        t_ttest.append(t1 - t0)

        # 2. Threshold comparison
        t0 = time.perf_counter()
        x_in_pos = _t_buf > threshold
        x_in_neg = _t_buf < -threshold
        t1 = time.perf_counter()
        t_threshold.append(t1 - t0)

        # Process each tail
        for x_in in [x_in_pos, x_in_neg]:
            if not np.any(x_in):
                continue

            # 3. np.where to get active indices
            t0 = time.perf_counter()
            idx = np.where(x_in)[0].astype(np.intp)
            n_active = len(idx)
            t1 = time.perf_counter()
            t_where.append(t1 - t0)

            # 4. Numba union-find CCL
            t0 = time.perf_counter()
            components = _fused_ccl(x_in, adj_row, adj_col, n_active, idx)
            t1 = time.perf_counter()
            t_ccl.append(t1 - t0)

            # 5. bincount sums
            t0 = time.perf_counter()
            sums = np.bincount(components, weights=_t_buf[idx])
            t1 = time.perf_counter()
            t_bincount.append(t1 - t0)

        # 6. Overhead (argmax etc.)
        t0 = time.perf_counter()
        # simulate the argmax + abs that happens in the actual loop
        _ = 0  # placeholder
        t1 = time.perf_counter()
        t_overhead.append(t1 - t0)

        t_total.append(time.perf_counter() - t0_total)

    # Report
    total_ms = np.sum(t_total) * 1000
    per_perm_ms = np.mean(t_total) * 1000

    components = {
        "ttest (signs@X + var)": np.sum(t_ttest) * 1000,
        "threshold (x > t)": np.sum(t_threshold) * 1000,
        "np.where": np.sum(t_where) * 1000,
        "_fused_ccl (Numba UF)": np.sum(t_ccl) * 1000,
        "bincount sums": np.sum(t_bincount) * 1000,
    }

    print(f"=== Profiling {n_perms} permutations ===")
    print(f"Total: {total_ms:.1f}ms ({per_perm_ms:.3f}ms/perm)")
    print()
    print(f"{'Component':<30} {'Total (ms)':>10} {'Per-perm (ms)':>13} {'%':>6}")
    print("-" * 65)
    for name, total in sorted(components.items(), key=lambda x: -x[1]):
        pct = 100 * total / total_ms
        per = total / n_perms
        print(f"{name:<30} {total:>10.1f} {per:>13.4f} {pct:>5.1f}%")
    accounted = sum(components.values())
    unaccounted = total_ms - accounted
    print(f"{'(unaccounted overhead)':<30} {unaccounted:>10.1f} {unaccounted/n_perms:>13.4f} {100*unaccounted/total_ms:>5.1f}%")
    print()

    # Also print stats on the number of active vertices per perm
    active_counts = []
    for order in orders:
        signs_1d = 2.0 * order - 1.0
        np.dot(signs_1d, X_2d, out=_mean_s)
        _mean_s /= n_samp
        np.multiply(_mean_s, _mean_s, out=_denom_sq)
        _denom_sq *= -n_samp
        _denom_sq += _sum_sq
        np.maximum(_denom_sq, 0, out=_denom_sq)
        np.sqrt(_denom_sq, out=_denom_sq)
        np.divide(_mean_s, _denom_sq, out=_t_buf)
        _t_buf *= _sqrt_n_nm1
        active_counts.append(np.sum(np.abs(_t_buf) > threshold))
    active_counts = np.array(active_counts)
    print(f"Active vertices per perm: mean={active_counts.mean():.0f}, "
          f"min={active_counts.min()}, max={active_counts.max()}, "
          f"total_tests={n_tests}")
    print(f"Edge list size: {len(adj_row):,}")
    print()

    return components


def main():
    print("Loading source-space data...")
    X, adjacency, n_subjects = load_source_data()
    print(f"Data shape: {X.shape}, adjacency: {adjacency.shape}")
    print()
    profile_perm_loop(X, adjacency, n_subjects, n_perms=256)


if __name__ == "__main__":
    main()
