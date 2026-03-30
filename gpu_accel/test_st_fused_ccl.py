#!/usr/bin/env python3
"""
Parity test: _st_fused_ccl vs BFS ground truth.

Verifies the new Numba union-find produces identical clustering results
to an independent BFS implementation (ground truth).

Note: The original _get_clusters_st_1step has a subtle bug where _reassign
modifies check1 in place during the nexts loop, causing stale snapshot values
in check1_d to orphan clusters. The union-find correctly computes the
transitive closure and matches the BFS ground truth exactly.

Tests:
  1. Small synthetic graph
  2. Varying density (many/few active vertices)
  3. Edge cases (no active, single vertex, all active)
  4. max_step > 1
  5. Performance comparison on medium graph
  6. Real fsaverage data from the benchmark
  7. End-to-end permutation test
"""

import time
from collections import defaultdict

import numpy as np
from scipy import sparse, stats

from mne.stats.cluster_level import (
    _get_clusters_st,
    _setup_adjacency,
    _st_fused_ccl,
)


def neighbors_to_csr(neighbors):
    """Convert neighbor list to CSR format."""
    n_src = len(neighbors)
    lengths = np.array([len(a) for a in neighbors])
    indptr = np.zeros(n_src + 1, dtype=np.intp)
    np.cumsum(lengths, out=indptr[1:])
    if n_src > 0 and np.sum(lengths) > 0:
        indices = np.concatenate(neighbors).astype(np.intp)
    else:
        indices = np.array([], dtype=np.intp)
    return indptr, indices


def bfs_clusters(x_in, neighbors, n_src, n_times, max_step=1):
    """Ground-truth clustering via BFS — independent of both implementations."""
    active = set(np.where(x_in)[0].tolist())
    visited = set()
    clusters = []
    for start in sorted(active):
        if start in visited:
            continue
        cluster = []
        queue = [start]
        visited.add(start)
        while queue:
            v = queue.pop(0)
            cluster.append(v)
            t = v // n_src
            s = v % n_src
            # Spatial neighbors at same time
            for n in neighbors[s]:
                flat_n = t * n_src + int(n)
                if flat_n in active and flat_n not in visited:
                    visited.add(flat_n)
                    queue.append(flat_n)
            # Temporal neighbors
            for step in range(1, max_step + 1):
                for dt in [-step, step]:
                    tn = t + dt
                    if 0 <= tn < n_times:
                        flat_n = tn * n_src + s
                        if flat_n in active and flat_n not in visited:
                            visited.add(flat_n)
                            queue.append(flat_n)
        clusters.append(sorted(cluster))
    return clusters


def cluster_sums_bfs(x_in, x, neighbors, n_src, n_times, max_step=1):
    """Get sorted cluster sums via BFS ground truth."""
    clusters = bfs_clusters(x_in, neighbors, n_src, n_times, max_step)
    if not clusters:
        return np.array([])
    return np.sort([np.sum(x[c]) for c in clusters])


def cluster_sums_uf(x_in, x, adj_indptr, adj_indices, n_src, n_times, max_step=1,
                     flat_map=None):
    """Get sorted cluster sums via union-find."""
    n_total = n_src * n_times
    if flat_map is None:
        flat_map = -np.ones(n_total, dtype=np.intp)
    act_idx = np.where(x_in)[0].astype(np.intp)
    n_act = len(act_idx)
    if n_act == 0:
        return np.array([])
    comps = _st_fused_ccl(
        act_idx, n_act, flat_map,
        adj_indptr, adj_indices, n_src, max_step,
    )
    return np.sort(np.bincount(comps, weights=x[act_idx]))


def cluster_sums_orig(x_in, x, neighbors, max_step=1):
    """Get sorted cluster sums via original _get_clusters_st."""
    clusters = _get_clusters_st(x_in, neighbors, max_step)
    if not clusters:
        return np.array([])
    all_idx = np.concatenate(clusters)
    lengths = np.array([len(c) for c in clusters])
    offsets = np.empty(len(clusters), dtype=np.intp)
    offsets[0] = 0
    np.cumsum(lengths[:-1], out=offsets[1:])
    return np.sort(np.add.reduceat(x[all_idx], offsets))


def make_ring_adjacency(n_src):
    """Create a ring adjacency."""
    row = list(range(n_src)) + list(range(n_src))
    col = [(i + 1) % n_src for i in range(n_src)] + [(i - 1) % n_src for i in range(n_src)]
    return sparse.csr_matrix(
        (np.ones(len(row)), (row, col)), shape=(n_src, n_src)
    )


def make_grid_adjacency(n_rows, n_cols):
    """Create a 2D grid adjacency."""
    n_src = n_rows * n_cols
    rows, cols = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            if c + 1 < n_cols:
                rows.extend([idx, idx + 1])
                cols.extend([idx + 1, idx])
            if r + 1 < n_rows:
                rows.extend([idx, idx + n_cols])
                cols.extend([idx + n_cols, idx])
    return sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_src, n_src)
    )


def setup_neighbors(adj):
    """Symmetrize and convert to neighbor lists + CSR."""
    adj = (adj + adj.T).tocsr()
    n_src = adj.shape[0]
    neighbors = [
        adj.indices[adj.indptr[i]:adj.indptr[i + 1]].astype(np.intp)
        for i in range(n_src)
    ]
    indptr, indices = neighbors_to_csr(neighbors)
    return neighbors, indptr, indices


def test_small_ring():
    """Test with a small ring graph, 3 time points."""
    print("Test 1: Small ring graph (10 vertices, 3 times)...")
    n_src = 10
    n_times = 3
    n_total = n_src * n_times

    neighbors, indptr, indices = setup_neighbors(make_ring_adjacency(n_src))

    np.random.seed(42)
    x = np.random.randn(n_total)
    x_in = np.abs(x) > 0.5

    sums_bfs = cluster_sums_bfs(x_in, x, neighbors, n_src, n_times)
    sums_uf = cluster_sums_uf(x_in, x, indptr, indices, n_src, n_times)

    assert len(sums_bfs) == len(sums_uf), \
        f"Count mismatch: BFS={len(sums_bfs)} UF={len(sums_uf)}"
    np.testing.assert_allclose(sums_bfs, sums_uf, atol=1e-10)
    print(f"  PASS: {len(sums_bfs)} clusters, sums match BFS ground truth")


def test_grid_varying_density():
    """Test with a grid graph at various densities."""
    print("Test 2: Grid graph, varying density...")
    n_rows, n_cols = 20, 30
    n_src = n_rows * n_cols
    n_times = 5
    n_total = n_src * n_times

    neighbors, indptr, indices = setup_neighbors(make_grid_adjacency(n_rows, n_cols))

    np.random.seed(123)
    x = np.random.randn(n_total)

    for threshold_pct in [0.1, 0.5, 0.9, 0.95, 0.99]:
        thresh_val = np.percentile(np.abs(x), threshold_pct * 100)
        x_in = np.abs(x) > thresh_val
        n_active = np.sum(x_in)

        sums_bfs = cluster_sums_bfs(x_in, x, neighbors, n_src, n_times)
        sums_uf = cluster_sums_uf(x_in, x, indptr, indices, n_src, n_times)

        assert len(sums_bfs) == len(sums_uf), \
            f"Density {threshold_pct}: count mismatch BFS={len(sums_bfs)} UF={len(sums_uf)}"
        if len(sums_bfs) > 0:
            np.testing.assert_allclose(sums_bfs, sums_uf, atol=1e-10)

        # Also check original — may have more clusters due to _reassign bug
        sums_orig = cluster_sums_orig(x_in, x, neighbors)
        bug_extra = len(sums_orig) - len(sums_bfs)
        suffix = f" (orig has +{bug_extra} extra)" if bug_extra > 0 else ""
        print(f"  density={1-threshold_pct:.0%} ({n_active}/{n_total} active): "
              f"{len(sums_bfs)} clusters OK{suffix}")


def test_edge_cases():
    """Test edge cases: no active, single active, all active."""
    print("Test 3: Edge cases...")
    n_src = 10
    n_times = 3
    n_total = n_src * n_times

    neighbors, indptr, indices = setup_neighbors(make_ring_adjacency(n_src))
    np.random.seed(0)
    x = np.random.randn(n_total)

    # No active vertices
    x_in = np.zeros(n_total, dtype=bool)
    sums_bfs = cluster_sums_bfs(x_in, x, neighbors, n_src, n_times)
    sums_uf = cluster_sums_uf(x_in, x, indptr, indices, n_src, n_times)
    assert len(sums_bfs) == 0 and len(sums_uf) == 0
    print("  No active: PASS")

    # Single active vertex
    x_in = np.zeros(n_total, dtype=bool)
    x_in[15] = True
    sums_bfs = cluster_sums_bfs(x_in, x, neighbors, n_src, n_times)
    sums_uf = cluster_sums_uf(x_in, x, indptr, indices, n_src, n_times)
    assert len(sums_bfs) == len(sums_uf) == 1
    np.testing.assert_allclose(sums_bfs, sums_uf, atol=1e-10)
    print("  Single active: PASS")

    # All active
    x_in = np.ones(n_total, dtype=bool)
    sums_bfs = cluster_sums_bfs(x_in, x, neighbors, n_src, n_times)
    sums_uf = cluster_sums_uf(x_in, x, indptr, indices, n_src, n_times)
    assert len(sums_bfs) == len(sums_uf)
    np.testing.assert_allclose(sums_bfs, sums_uf, atol=1e-10)
    print(f"  All active: PASS ({len(sums_bfs)} clusters)")

    # Two isolated active vertices
    x_in = np.zeros(n_total, dtype=bool)
    x_in[0] = True
    x_in[n_src * 2 + 5] = True
    sums_bfs = cluster_sums_bfs(x_in, x, neighbors, n_src, n_times)
    sums_uf = cluster_sums_uf(x_in, x, indptr, indices, n_src, n_times)
    assert len(sums_bfs) == len(sums_uf) == 2
    np.testing.assert_allclose(sums_bfs, sums_uf, atol=1e-10)
    print("  Two isolated: PASS")


def test_max_step():
    """Test with max_step > 1."""
    print("Test 4: max_step=2...")
    n_src = 10
    n_times = 5
    n_total = n_src * n_times

    neighbors, indptr, indices = setup_neighbors(make_ring_adjacency(n_src))

    np.random.seed(99)
    x = np.random.randn(n_total)
    x_in = np.abs(x) > 0.3

    sums_bfs = cluster_sums_bfs(x_in, x, neighbors, n_src, n_times, max_step=2)
    sums_uf = cluster_sums_uf(x_in, x, indptr, indices, n_src, n_times, max_step=2)

    assert len(sums_bfs) == len(sums_uf), \
        f"Count mismatch: BFS={len(sums_bfs)} UF={len(sums_uf)}"
    if len(sums_bfs) > 0:
        np.testing.assert_allclose(sums_bfs, sums_uf, atol=1e-10)
    print(f"  PASS: {len(sums_bfs)} clusters, sums match")


def test_perf_synthetic():
    """Performance comparison on a medium-sized synthetic graph."""
    print("\nTest 5: Performance (200x100 grid, 15 times)...")
    n_rows, n_cols = 200, 100
    n_src = n_rows * n_cols
    n_times = 15
    n_total = n_src * n_times

    neighbors, indptr, indices = setup_neighbors(
        make_grid_adjacency(n_rows, n_cols)
    )

    np.random.seed(0)
    x = np.random.randn(n_total)
    x_in = np.abs(x) > 1.5  # ~13% active

    n_active = np.sum(x_in)
    print(f"  n_src={n_src}, n_times={n_times}, n_total={n_total}, "
          f"n_active={n_active} ({100*n_active/n_total:.1f}%)")

    # Pre-allocate flat_map buffer for performance
    flat_map = -np.ones(n_total, dtype=np.intp)

    # Warmup Numba JIT
    _ = cluster_sums_uf(x_in, x, indptr, indices, n_src, n_times,
                         flat_map=flat_map)

    n_iters = 20

    # Original
    t0 = time.perf_counter()
    for _ in range(n_iters):
        sums_orig = cluster_sums_orig(x_in, x, neighbors)
    t_orig = (time.perf_counter() - t0) / n_iters

    # New (with pre-allocated buffers)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        sums_new = cluster_sums_uf(x_in, x, indptr, indices, n_src, n_times,
                                    flat_map=flat_map)
    t_new = (time.perf_counter() - t0) / n_iters

    # Verify against BFS
    sums_bfs = cluster_sums_bfs(x_in, x, neighbors, n_src, n_times)
    np.testing.assert_allclose(sums_bfs, sums_new, atol=1e-10)

    print(f"  Original (_get_clusters_st): {t_orig*1000:.2f} ms/call")
    print(f"  New (_st_fused_ccl):         {t_new*1000:.2f} ms/call")
    print(f"  Speedup: {t_orig/t_new:.1f}x")
    print(f"  Sums match BFS ground truth: YES")


def test_fsaverage():
    """Test with real fsaverage data from the benchmark."""
    print("\nTest 6: Real fsaverage data...")
    try:
        import mne
        from mne.datasets import sample
        from mne.minimum_norm import apply_inverse, read_inverse_operator
        from mne.stats.parametric import ttest_1samp_no_p
    except ImportError:
        print("  SKIP: MNE not fully available")
        return

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

    n_vertices_sample, n_times_data = condition1.data.shape
    n_subjects = 7

    np.random.seed(0)
    X = np.random.randn(n_vertices_sample, n_times_data, n_subjects, 2) * 10
    X[:, :, :, 0] += condition1.data[:, :, np.newaxis]
    X[:, :, :, 1] += condition2.data[:, :, np.newaxis]

    n_vertices_fsave = morph_mat.shape[0]
    X = morph_mat.dot(X.reshape(n_vertices_sample, -1))
    X = X.reshape(n_vertices_fsave, n_times_data, n_subjects, 2)
    X = np.abs(X)
    X = X[:, :, :, 0] - X[:, :, :, 1]
    X = np.transpose(X, [2, 1, 0])

    adjacency = mne.spatial_src_adjacency(src)
    n_samp, n_times_d, n_vertices = X.shape
    X_2d = X.reshape(n_samp, -1)
    n_tests = X_2d.shape[1]
    n_src = adjacency.shape[0]

    adj_setup = _setup_adjacency(adjacency, n_tests, n_times_d)
    assert isinstance(adj_setup, list)
    neighbors = adj_setup
    indptr, indices = neighbors_to_csr(neighbors)

    t_obs = ttest_1samp_no_p(X_2d)
    df = n_subjects - 1
    threshold = stats.distributions.t.ppf(1 - 0.001 / 2, df=df)

    print(f"  Data: {n_samp} subjects, {n_times_d} times, {n_vertices} vertices")
    print(f"  n_tests={n_tests}, n_src={n_src}")

    # Pre-allocate flat_map buffer
    flat_map = -np.ones(n_tests, dtype=np.intp)

    # Warmup (use sparse x_in to avoid slow all-active warmup)
    _dummy = np.zeros(n_tests, dtype=bool)
    _dummy[0] = True
    _ = cluster_sums_uf(
        _dummy, t_obs, indptr, indices, n_src, n_times_d,
        flat_map=flat_map,
    )
    del _dummy

    # Test with random permutations
    rng = np.random.RandomState(42)
    n_perm_test = 50
    _sum_sq = np.sum(X_2d ** 2, axis=0)
    _sqrt_n_nm1 = np.sqrt(n_samp * (n_samp - 1))
    _mean_s = np.empty(n_tests, dtype=np.float64)
    _denom_sq = np.empty(n_tests, dtype=np.float64)
    _t_buf = np.empty(n_tests, dtype=np.float64)

    t_orig_total = 0
    t_new_total = 0
    n_orig_extra = 0

    print(f"  Testing {n_perm_test} random permutations...")
    for pi in range(n_perm_test):
        order = rng.choice([False, True], size=n_samp)
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

        for sign in [1, -1]:
            x_in = _t_buf > threshold if sign == 1 else _t_buf < -threshold
            if not np.any(x_in):
                continue

            t0 = time.perf_counter()
            sums_orig = cluster_sums_orig(x_in, _t_buf, neighbors)
            t_orig_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            sums_uf = cluster_sums_uf(
                x_in, _t_buf, indptr, indices, n_src, n_times_d,
                flat_map=flat_map,
            )
            t_new_total += time.perf_counter() - t0

            # UF should have <= clusters than orig (orig over-counts)
            if len(sums_orig) > len(sums_uf):
                n_orig_extra += len(sums_orig) - len(sums_uf)

            # Key check: max cluster sum should be very close
            # (UF may merge clusters, changing individual sums but
            #  the max should be >= orig max since merging only increases)
            if len(sums_uf) > 0 and len(sums_orig) > 0:
                max_uf = np.max(np.abs(sums_uf))
                max_orig = np.max(np.abs(sums_orig))
                assert max_uf >= max_orig - 1e-10, \
                    f"Perm {pi}: UF max {max_uf} < orig max {max_orig}"

    print(f"  All {n_perm_test} permutations PASS")
    print(f"  Original over-counted by {n_orig_extra} clusters total across all perms")
    print(f"  Original: {t_orig_total*1000:.1f} ms ({t_orig_total/n_perm_test*1000:.2f} ms/perm)")
    print(f"  New:      {t_new_total*1000:.1f} ms ({t_new_total/n_perm_test*1000:.2f} ms/perm)")
    print(f"  CCL speedup: {t_orig_total/t_new_total:.1f}x")


def test_e2e_permutation_test():
    """End-to-end permutation test with the optimization active."""
    print("\nTest 7: End-to-end permutation test...")
    try:
        from mne.stats import spatio_temporal_cluster_1samp_test
    except ImportError:
        print("  SKIP: MNE not fully available")
        return

    n_src = 100
    n_times = 5
    n_subjects = 10
    n_perms = 50
    adj = make_grid_adjacency(10, 10)

    np.random.seed(42)
    X = np.random.randn(n_subjects, n_times, n_src) * 2
    X[:, 2:4, 40:60] += 3.0

    df = n_subjects - 1
    threshold = stats.distributions.t.ppf(1 - 0.05 / 2, df=df)

    T_obs, clusters, pv, H0 = spatio_temporal_cluster_1samp_test(
        X, adjacency=adj, threshold=threshold,
        n_permutations=n_perms, seed=42, verbose=False,
    )

    print(f"  {len(clusters)} clusters, H0 shape: {H0.shape}")
    print(f"  min p-value: {pv.min():.4f}")
    print(f"  H0 range: [{H0.min():.2f}, {H0.max():.2f}]")
    assert len(clusters) > 0
    # H0 includes the original observation + permutations
    assert H0.shape[0] >= n_perms
    print("  E2E test PASS")


if __name__ == "__main__":
    test_small_ring()
    test_grid_varying_density()
    test_edge_cases()
    test_max_step()
    test_perf_synthetic()
    test_fsaverage()
    test_e2e_permutation_test()
    print("\n=== ALL TESTS PASSED ===")
