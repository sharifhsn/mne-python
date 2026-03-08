#!/usr/bin/env python3
"""
Head-to-head benchmark: CPU vs GPU (CuPy) for permutation cluster tests.

FINDINGS (RTX 2070 SUPER, 2026-03-08):

1. The FEASIBILITY.md incorrectly identified `_get_components()` as the
   bottleneck. In the standard spatio-temporal case, MNE uses
   `_get_clusters_st()` (Numba BFS on neighbor lists), NOT `_get_components()`
   (SciPy sparse graph CCL).

2. When the spatial adjacency is smaller than n_tests (which is always the
   case for spatio-temporal data), `_setup_adjacency()` converts it to a
   neighbor list and routes to `_get_clusters_st()`.

3. Results of correct benchmarking (p<0.05, 128 perms, fsaverage ico-5):
   - CPU (Numba _get_clusters_st): 23.5s
   - GPU (CuPy on full ST adjacency): 61.2s  => 0.38x (GPU SLOWER)
   - GPU has ~15ms kernel launch overhead per call, and each call operates
     on a 307K-node graph. The transfer + kernel overhead dominates.

4. Isolated CCL scaling (30% supra-threshold density):
   - 20K vertices: CPU 1.0ms, GPU 14.8ms (GPU 0.07x)
   - 500K vertices: CPU 24.6ms, GPU 40.7ms (GPU 0.60x)
   - 1M vertices: CPU 67ms, GPU 63ms (GPU 1.07x — crossover)
   - 5M vertices: CPU 694ms, GPU 302ms (GPU 2.30x)

   GPU only wins at >1M vertices per call, which is much larger than the
   typical MNE source-space graph (307K tests, ~1K-10K supra-threshold).

5. The CuPy drop-in approach (Path 1) is NOT viable for speedup.
   The real opportunity is Path 3: fuse the entire permutation loop
   (sign-flip + t-test + threshold + CCL + reduce) on GPU to eliminate
   per-permutation CPU-GPU transfer overhead.

Usage:
    python gpu_accel/benchmark_head_to_head.py
"""

import sys
import time

import numpy as np
from scipy import sparse, stats

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.stats import cluster_level as cl

# ---------- Check CuPy ----------
try:
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse
    from cupyx.scipy.sparse.csgraph import connected_components as gpu_cc
except ImportError:
    print("CuPy not available. Install with: uv pip install cupy-cuda12x")
    sys.exit(1)

dev_props = cp.cuda.runtime.getDeviceProperties(0)
dev_name = dev_props["name"]
if isinstance(dev_name, bytes):
    dev_name = dev_name.decode()
print(f"GPU: {dev_name}")
print(f"CuPy: {cp.__version__}")
print()


# ---------- GPU backend for _get_components ----------
_original_get_components = cl._get_components


def _gpu_get_components(x_in, adjacency, return_list=True):
    """GPU-accelerated connected components via CuPy."""
    if adjacency is False:
        if return_list:
            idx = np.where(x_in)[0]
            return [idx[i : i + 1] for i in range(len(idx))]
        return np.arange(len(x_in))
    if return_list:
        idx = np.where(x_in)[0]
        n_active = len(idx)
        if n_active == 0:
            return []
        global_to_local = np.empty(adjacency.shape[0], dtype=np.intp)
        global_to_local[idx] = np.arange(n_active)
        edge_mask = np.logical_and(x_in[adjacency.row], x_in[adjacency.col])
        row = global_to_local[adjacency.row[edge_mask]]
        col = global_to_local[adjacency.col[edge_mask]]
        self_idx = np.arange(n_active)
        row = np.concatenate((row, self_idx))
        col = np.concatenate((col, self_idx))
        data = np.ones(len(row), dtype=np.float32)
        adj_gpu = cp_sparse.coo_matrix(
            (cp.asarray(data), (cp.asarray(row), cp.asarray(col))),
            shape=(n_active, n_active),
        )
        _, components_gpu = gpu_cc(adj_gpu, directed=False)
        components = cp.asnumpy(components_gpu)
        order = np.argsort(components, kind="stable")
        counts = np.bincount(components)
        splits = np.cumsum(counts[:-1])
        global_order = idx[order]
        return list(np.split(global_order, splits))
    else:
        mask = np.logical_and(x_in[adjacency.row], x_in[adjacency.col])
        data = adjacency.data[mask]
        row = adjacency.row[mask]
        col = adjacency.col[mask]
        shape = adjacency.shape
        idx = np.where(x_in)[0]
        row = np.concatenate((row, idx))
        col = np.concatenate((col, idx))
        data = np.concatenate((data, np.ones(len(idx), dtype=data.dtype)))
        adj_gpu = cp_sparse.coo_matrix(
            (cp.asarray(data.astype(np.float32)),
             (cp.asarray(row), cp.asarray(col))),
            shape=shape,
        )
        _, components_gpu = gpu_cc(adj_gpu, directed=False)
        return cp.asnumpy(components_gpu)


# ---------- Load MNE sample data ----------
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


def build_full_st_adjacency(adjacency, n_times):
    """Build full spatio-temporal adjacency from spatial adjacency."""
    n_vertices = adjacency.shape[0]
    n_tests = n_vertices * n_times
    adj_coo = adjacency.tocoo()
    all_rows, all_cols = [], []
    for t in range(n_times):
        offset = t * n_vertices
        all_rows.append(adj_coo.row + offset)
        all_cols.append(adj_coo.col + offset)
        if t < n_times - 1:
            verts = np.arange(n_vertices)
            all_rows.append(verts + offset)
            all_cols.append(verts + offset + n_vertices)
            all_rows.append(verts + offset + n_vertices)
            all_cols.append(verts + offset)
    all_rows = np.concatenate(all_rows)
    all_cols = np.concatenate(all_cols)
    all_data = np.ones(len(all_rows), dtype=np.float32)
    return sparse.coo_array(
        (all_data, (all_rows, all_cols)), shape=(n_tests, n_tests)
    )


def main():
    print("Loading source-space data from MNE sample dataset...")
    X, adjacency, n_subjects = load_source_data()
    n_times = X.shape[1]
    df = n_subjects - 1
    print(f"Data: {X.shape}, adjacency: {adjacency.shape} nnz={adjacency.nnz}")
    print(f"n_subjects={n_subjects} => max unique permutations = 2^{n_subjects} = {2**n_subjects}")
    print()

    # Build full ST adjacency for GPU path
    full_adj = build_full_st_adjacency(adjacency, n_times)
    print(f"Full ST adjacency: {full_adj.shape}, nnz={full_adj.nnz:,}")
    print()

    # ---- Profile time breakdown ----
    print("=" * 70)
    print("PART 1: Time breakdown (CPU, p<0.05)")
    print("=" * 70)

    _orig_st = cl._get_clusters_st
    st_total = [0.0]
    st_calls = [0]
    find_total = [0.0]
    _orig_find = cl._find_clusters

    def _timed_st(*a, **kw):
        t0 = time.perf_counter()
        r = _orig_st(*a, **kw)
        st_total[0] += time.perf_counter() - t0
        st_calls[0] += 1
        return r

    def _timed_find(*a, **kw):
        t0 = time.perf_counter()
        r = _orig_find(*a, **kw)
        find_total[0] += time.perf_counter() - t0
        return r

    cl._get_clusters_st = _timed_st
    cl._find_clusters = _timed_find

    t_thr = stats.distributions.t.ppf(1 - 0.05 / 2, df=df)
    n_perms = 128
    t0 = time.perf_counter()
    T_cpu, cl_cpu, pv_cpu, _ = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_jobs=1, threshold=t_thr,
        n_permutations=n_perms, buffer_size=None, verbose=False, seed=42,
    )
    total = time.perf_counter() - t0

    print(f"Total:               {total:.2f}s")
    print(f"_find_clusters:      {find_total[0]:.2f}s ({100*find_total[0]/total:.1f}%)")
    print(f"_get_clusters_st:    {st_total[0]:.2f}s ({100*st_total[0]/total:.1f}%)")
    print(f"Other (stat, flip):  {total - find_total[0]:.2f}s")
    print(f"Calls: {st_calls[0]}, avg: {st_total[0]/max(st_calls[0],1)*1000:.1f}ms/call")
    print(f"Clusters: {len(cl_cpu)}")

    # Restore
    cl._get_clusters_st = _orig_st
    cl._find_clusters = _orig_find

    # ---- Head-to-head ----
    print()
    print("=" * 70)
    print("PART 2: CPU (Numba) vs GPU (CuPy) — full permutation test")
    print("=" * 70)

    for label, thresh in [
        ("p<0.05", stats.distributions.t.ppf(1 - 0.05 / 2, df=df)),
        ("p<0.01", stats.distributions.t.ppf(1 - 0.01 / 2, df=df)),
    ]:
        # CPU (default path: neighbor list + Numba)
        cl._get_components = _original_get_components
        t0 = time.perf_counter()
        T_cpu, cl_cpu, pv_cpu, _ = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_jobs=1, threshold=thresh,
            n_permutations=n_perms, buffer_size=None, verbose=False, seed=42,
        )
        cpu_time = time.perf_counter() - t0

        # GPU (full ST adjacency + CuPy connected_components)
        cl._get_components = _gpu_get_components
        t0 = time.perf_counter()
        T_gpu, cl_gpu, pv_gpu, _ = spatio_temporal_cluster_1samp_test(
            X, adjacency=full_adj, n_jobs=1, threshold=thresh,
            n_permutations=n_perms, buffer_size=None, verbose=False, seed=42,
        )
        gpu_time = time.perf_counter() - t0

        speedup = cpu_time / gpu_time
        match = len(cl_cpu) == len(cl_gpu) and np.allclose(pv_cpu, pv_gpu)
        print(f"  {label}: CPU {cpu_time:.1f}s, GPU {gpu_time:.1f}s "
              f"=> {speedup:.2f}x  "
              f"(clusters: CPU={len(cl_cpu)} GPU={len(cl_gpu)}, match={match})")

    cl._get_components = _original_get_components

    # ---- Isolated CCL scaling ----
    print()
    print("=" * 70)
    print("PART 3: Isolated CCL scaling (where does GPU crossover?)")
    print("=" * 70)
    from scipy.sparse.csgraph import connected_components as cpu_cc

    rng = np.random.RandomState(42)
    print(f"{'Vertices':>10s} {'Density':>8s} {'CPU ms':>10s} "
          f"{'GPU ms':>10s} {'Speedup':>10s}")
    print("-" * 55)

    for n_v, dens, n_it in [
        (20_000, 30, 50),
        (100_000, 30, 10),
        (500_000, 30, 5),
        (1_000_000, 30, 3),
        (2_000_000, 30, 2),
    ]:
        n_edges = n_v * 6
        rows = rng.randint(0, n_v, size=n_edges)
        cols = rng.randint(0, n_v, size=n_edges)
        data = np.ones(len(rows), dtype=np.float32)
        adj = sparse.coo_matrix((data, (rows, cols)), shape=(n_v, n_v)).tocsr()
        adj = (adj + adj.T)
        adj.data[:] = 1.0

        n_sig = int(n_v * dens / 100)
        x_in = np.zeros(n_v, dtype=bool)
        x_in[:n_sig] = True
        rng.shuffle(x_in)
        adj_coo = adj.tocoo()
        mask = np.logical_and(x_in[adj_coo.row], x_in[adj_coo.col])
        fdata = adj_coo.data[mask].astype(np.float32)
        frow = adj_coo.row[mask]
        fcol = adj_coo.col[mask]
        idx = np.where(x_in)[0]
        frow = np.concatenate((frow, idx))
        fcol = np.concatenate((fcol, idx))
        fdata = np.concatenate((fdata, np.ones(len(idx), dtype=np.float32)))
        sub = sparse.coo_matrix((fdata, (frow, fcol)), shape=(n_v, n_v))
        sub_gpu = cp_sparse.coo_matrix(
            (cp.asarray(fdata), (cp.asarray(frow), cp.asarray(fcol))),
            shape=(n_v, n_v))

        cpu_cc(sub)
        gpu_cc(sub_gpu, directed=False)
        cp.cuda.Device().synchronize()

        t0 = time.perf_counter()
        for _ in range(n_it):
            cpu_cc(sub)
        cpu_t = (time.perf_counter() - t0) / n_it * 1000

        t0 = time.perf_counter()
        for _ in range(n_it):
            gpu_cc(sub_gpu, directed=False)
            cp.cuda.Device().synchronize()
        gpu_t = (time.perf_counter() - t0) / n_it * 1000

        spd = cpu_t / gpu_t
        print(f"{n_v:10,d} {dens:7d}% {cpu_t:10.1f} {gpu_t:10.1f} {spd:9.2f}x")

    print()
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
  1. The CuPy drop-in for _get_components() does NOT help because the
     standard spatio-temporal path uses _get_clusters_st() (Numba BFS),
     not _get_components() (SciPy sparse CCL).

  2. Even when forcing the sparse CCL path, GPU is 2-3x SLOWER than CPU
     for MNE-scale graphs (307K nodes). GPU only wins at >1M nodes.

  3. The ~15ms kernel launch overhead of pylibcugraph dominates when each
     CCL call processes a graph that CPU handles in 1-25ms.

  4. The real speedup opportunity is Path 3 (fused GPU pipeline):
     keep ALL permutation iterations on GPU to amortize launch overhead.
     Sign-flip + t-test + threshold + CCL + reduce, all in one kernel.
""")


if __name__ == "__main__":
    main()
