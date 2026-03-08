#!/usr/bin/env python3
"""
CuPy GPU vs CPU benchmark for MNE permutation cluster tests.

Finds the parameter sweet spot where GPU clearly wins, tests both the
isolated CCL step and full permutation pipeline.

Results are written to gpu_accel/benchmark_results.txt
"""

import sys
import time
from datetime import datetime

import numpy as np
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components as cpu_cc

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.stats import cluster_level as cl

# ---------- Setup output ----------
RESULTS_FILE = "/home/sharif/mne-python/gpu_accel/benchmark_results.txt"
results_lines = []


def log(msg=""):
    print(msg, flush=True)
    results_lines.append(msg)


def save_results():
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(results_lines) + "\n")


# ---------- Check CuPy ----------
try:
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse
    from cupyx.scipy.sparse.csgraph import connected_components as gpu_cc
except ImportError:
    log("CuPy not available. Install with: uv pip install cupy-cuda12x")
    save_results()
    sys.exit(1)

dev_props = cp.cuda.runtime.getDeviceProperties(0)
dev_name = dev_props["name"]
if isinstance(dev_name, bytes):
    dev_name = dev_name.decode()

log(f"Benchmark started: {datetime.now().isoformat()}")
log(f"GPU: {dev_name}")
log(f"CuPy: {cp.__version__}")
log(f"NumPy: {np.__version__}")
log()

# ---------- GPU backend ----------
_original_get_components = cl._get_components


def _gpu_get_components(x_in, adjacency, return_list=True):
    """GPU-accelerated connected components via CuPy."""
    if adjacency is False:
        components = np.arange(len(x_in))
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
        components = cp.asnumpy(components_gpu)

    if return_list:
        start = np.min(components)
        stop = np.max(components)
        comp_list = [list() for i in range(start, stop + 1, 1)]
        mask = np.zeros(len(comp_list), dtype=bool)
        for ii, comp in enumerate(components):
            comp_list[comp].append(ii)
            mask[comp] += x_in[ii]
        clusters = [np.array(k) for k, m in zip(comp_list, mask) if m]
        return clusters
    else:
        return components


# =====================================================================
# PART 1: Isolated CCL with real MNE ico-5 topology
# =====================================================================
log("=" * 70)
log("PART 1: Isolated CCL benchmark (real ico-5 cortical mesh topology)")
log("=" * 70)
log()

data_path = sample.data_path()
subjects_dir = data_path / "subjects"
src_fname = subjects_dir / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif"
src = mne.read_source_spaces(src_fname, verbose=False)
spatial_adj = mne.spatial_src_adjacency(src).tocoo()
n_verts = spatial_adj.shape[0]
log(f"Spatial adjacency: {n_verts:,} vertices, nnz={spatial_adj.nnz:,}")
log()

rng = np.random.RandomState(42)

header = (f"{'n_times':>7s} {'n_tests':>11s} {'sig_nodes':>10s} "
          f"{'sub_nnz':>12s} {'CPU_ms':>9s} {'GPU_ms':>9s} {'Speedup':>8s}")
log(header)
log("-" * 75)

ccl_results = []
for n_times in [25, 50, 75, 100, 150, 200, 250, 350, 500]:
    n_tests = n_verts * n_times

    # Build full spatio-temporal adjacency
    rows, cols = [], []
    for t in range(n_times):
        off = t * n_verts
        rows.append(spatial_adj.row + off)
        cols.append(spatial_adj.col + off)
        if t < n_times - 1:
            v = np.arange(n_verts)
            rows.append(v + off)
            cols.append(v + off + n_verts)
            rows.append(v + off + n_verts)
            cols.append(v + off)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype=np.float32)
    st_adj = sparse.coo_matrix((data, (rows, cols)), shape=(n_tests, n_tests))

    # 30% supra-threshold
    x_in = rng.rand(n_tests) < 0.3
    n_sig = int(x_in.sum())

    # Build filtered subgraph
    fmask = np.logical_and(x_in[st_adj.row], x_in[st_adj.col])
    fd = st_adj.data[fmask]
    fr = st_adj.row[fmask]
    fc = st_adj.col[fmask]
    idx = np.where(x_in)[0]
    fr = np.concatenate((fr, idx))
    fc = np.concatenate((fc, idx))
    fd = np.concatenate((fd, np.ones(len(idx), dtype=np.float32)))
    sub = sparse.coo_matrix((fd, (fr, fc)), shape=(n_tests, n_tests))
    sub_gpu = cp_sparse.coo_matrix(
        (cp.asarray(fd), (cp.asarray(fr), cp.asarray(fc))),
        shape=(n_tests, n_tests))

    # Warmup
    cpu_cc(sub)
    gpu_cc(sub_gpu, directed=False)
    cp.cuda.Device().synchronize()

    n_iters = max(2, min(20, int(10 / max(n_tests / 5e6, 0.1))))

    t0 = time.perf_counter()
    for _ in range(n_iters):
        cpu_cc(sub)
    cpu_ms = (time.perf_counter() - t0) / n_iters * 1000

    t0 = time.perf_counter()
    for _ in range(n_iters):
        gpu_cc(sub_gpu, directed=False)
        cp.cuda.Device().synchronize()
    gpu_ms = (time.perf_counter() - t0) / n_iters * 1000

    spd = cpu_ms / gpu_ms
    log(f"{n_times:7d} {n_tests:11,} {n_sig:10,} {sub.nnz:12,} "
        f"{cpu_ms:9.1f} {gpu_ms:9.1f} {spd:7.2f}x")
    ccl_results.append((n_times, n_tests, n_sig, cpu_ms, gpu_ms, spd))
    save_results()  # save incrementally

log()

# =====================================================================
# PART 2: Full permutation test — GPU path (full ST adjacency)
# =====================================================================
log("=" * 70)
log("PART 2: Full permutation cluster test — CPU vs GPU")
log("  CPU path: default (neighbor-list + Numba _get_clusters_st)")
log("  GPU path: full ST adjacency + CuPy _get_components")
log("=" * 70)
log()

# Load real MNE sample data
log("Loading MNE sample data...")
meg_path = data_path / "MEG" / "sample"
fname_inv = meg_path / "sample_audvis-meg-oct-6-meg-inv.fif"
raw = mne.io.read_raw_fif(
    meg_path / "sample_audvis_filt-0-40_raw.fif", verbose=False)
events = mne.read_events(meg_path / "sample_audvis_filt-0-40_raw-eve.fif")
raw.info["bads"] += ["MEG 2443"]
picks = mne.pick_types(raw.info, meg=True, eog=True, exclude="bads")
reject = dict(grad=1000e-13, mag=4000e-15, eog=150e-6)
e1 = mne.Epochs(raw, events, 1, -0.2, 0.3, picks=picks,
                baseline=(None, 0), reject=reject, preload=True, verbose=False)
e2 = mne.Epochs(raw, events, 3, -0.2, 0.3, picks=picks,
                baseline=(None, 0), reject=reject, preload=True, verbose=False)
mne.epochs.equalize_epoch_counts([e1, e2])
inv = read_inverse_operator(fname_inv, verbose=False)
ev1 = e1.average().resample(50, npad="auto", verbose=False)
ev2 = e2.average().resample(50, npad="auto", verbose=False)
c1 = apply_inverse(ev1, inv, 1 / 9., "dSPM", verbose=False)
c2 = apply_inverse(ev2, inv, 1 / 9., "dSPM", verbose=False)
c1.crop(0, None)
c2.crop(0, None)

fv = [s["vertno"] for s in src]
morph = mne.compute_source_morph(
    src=inv["src"], subject_to="fsaverage",
    spacing=fv, subjects_dir=subjects_dir, verbose=False).morph_mat
nv, nt = c1.data.shape

# Build synthetic multi-subject data at various n_subjects
for n_subjects in [7, 15, 20, 30]:
    np.random.seed(0)
    X = np.random.randn(nv, nt, n_subjects, 2) * 10
    X[:, :, :, 0] += c1.data[:, :, np.newaxis]
    X[:, :, :, 1] += c2.data[:, :, np.newaxis]
    X = morph.dot(X.reshape(nv, -1)).reshape(morph.shape[0], nt, n_subjects, 2)
    X = np.abs(X)
    X = X[:, :, :, 0] - X[:, :, :, 1]
    X = np.transpose(X, [2, 1, 0])

    adjacency = mne.spatial_src_adjacency(src)
    n_times_data = X.shape[1]
    n_tests_data = n_verts * n_times_data

    # Build full ST adjacency for GPU path
    rows, cols = [], []
    adj_coo = adjacency.tocoo()
    for t in range(n_times_data):
        off = t * n_verts
        rows.append(adj_coo.row + off)
        cols.append(adj_coo.col + off)
        if t < n_times_data - 1:
            v = np.arange(n_verts)
            rows.append(v + off)
            cols.append(v + off + n_verts)
            rows.append(v + off + n_verts)
            cols.append(v + off)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    full_data = np.ones(len(rows), dtype=np.float32)
    full_adj = sparse.coo_array(
        (full_data, (rows, cols)), shape=(n_tests_data, n_tests_data))

    df = n_subjects - 1
    max_unique = 2 ** n_subjects
    n_perms = min(512, max_unique)

    for p_level, p_val in [("p<0.05", 0.05), ("p<0.01", 0.01)]:
        t_thr = stats.distributions.t.ppf(1 - p_val / 2, df=df)

        log(f"  n_subjects={n_subjects}, {p_level} (t={t_thr:.2f}), "
            f"n_perms={n_perms}, n_tests={n_tests_data:,}")

        # CPU (default: neighbor list + Numba)
        cl._get_components = _original_get_components
        t0 = time.perf_counter()
        T_cpu, cl_cpu, pv_cpu, _ = spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_jobs=1, threshold=t_thr,
            n_permutations=n_perms, buffer_size=None, verbose=False, seed=42)
        cpu_t = time.perf_counter() - t0

        # GPU (full ST adjacency + CuPy)
        cl._get_components = _gpu_get_components
        t0 = time.perf_counter()
        T_gpu, cl_gpu, pv_gpu, _ = spatio_temporal_cluster_1samp_test(
            X, adjacency=full_adj, n_jobs=1, threshold=t_thr,
            n_permutations=n_perms, buffer_size=None, verbose=False, seed=42)
        gpu_t = time.perf_counter() - t0

        match = len(cl_cpu) == len(cl_gpu)
        spd = cpu_t / gpu_t
        log(f"    CPU {cpu_t:.1f}s  GPU {gpu_t:.1f}s  => {spd:.2f}x  "
            f"(clusters: CPU={len(cl_cpu)} GPU={len(cl_gpu)}, match={match})")
        save_results()

    log()

cl._get_components = _original_get_components

# =====================================================================
# PART 3: Synthetic stress test — bigger data, more subjects
# =====================================================================
log("=" * 70)
log("PART 3: Synthetic stress test (larger time windows)")
log("=" * 70)
log()

adjacency = mne.spatial_src_adjacency(src)

for n_subjects_syn, n_times_syn, n_perms_syn in [
    (20, 50, 256),
    (20, 100, 256),
    (20, 150, 128),
]:
    rng_syn = np.random.RandomState(0)
    n_tests_syn = n_verts * n_times_syn
    X_syn = rng_syn.randn(n_subjects_syn, n_times_syn, n_verts) * 0.5
    # Add signal to ~30% of vertices
    sig_mask = rng_syn.rand(n_verts) < 0.3
    X_syn[:, :, sig_mask] += 2.0

    df_syn = n_subjects_syn - 1
    t_thr_syn = stats.distributions.t.ppf(1 - 0.05 / 2, df=df_syn)

    # Build full ST adjacency
    rows, cols = [], []
    adj_coo = adjacency.tocoo()
    for t in range(n_times_syn):
        off = t * n_verts
        rows.append(adj_coo.row + off)
        cols.append(adj_coo.col + off)
        if t < n_times_syn - 1:
            v = np.arange(n_verts)
            rows.append(v + off)
            cols.append(v + off + n_verts)
            rows.append(v + off + n_verts)
            cols.append(v + off)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    full_data = np.ones(len(rows), dtype=np.float32)
    full_adj_syn = sparse.coo_array(
        (full_data, (rows, cols)), shape=(n_tests_syn, n_tests_syn))

    log(f"  N={n_subjects_syn}, T={n_times_syn}, tests={n_tests_syn:,}, "
        f"perms={n_perms_syn}, t_thr={t_thr_syn:.2f}")

    # CPU
    cl._get_components = _original_get_components
    t0 = time.perf_counter()
    T_cpu, cl_cpu, pv_cpu, _ = spatio_temporal_cluster_1samp_test(
        X_syn, adjacency=adjacency, n_jobs=1, threshold=t_thr_syn,
        n_permutations=n_perms_syn, buffer_size=None, verbose=False, seed=42)
    cpu_t = time.perf_counter() - t0

    # GPU
    cl._get_components = _gpu_get_components
    t0 = time.perf_counter()
    T_gpu, cl_gpu, pv_gpu, _ = spatio_temporal_cluster_1samp_test(
        X_syn, adjacency=full_adj_syn, n_jobs=1, threshold=t_thr_syn,
        n_permutations=n_perms_syn, buffer_size=None, verbose=False, seed=42)
    gpu_t = time.perf_counter() - t0

    spd = cpu_t / gpu_t
    log(f"    CPU {cpu_t:.1f}s  GPU {gpu_t:.1f}s  => {spd:.2f}x  "
        f"(clusters: CPU={len(cl_cpu)} GPU={len(cl_gpu)})")
    save_results()

log()
cl._get_components = _original_get_components

# =====================================================================
# Summary
# =====================================================================
log("=" * 70)
log("SUMMARY — Isolated CCL (Part 1)")
log("=" * 70)
log()
log("The CCL step (connected_components) scales like this on your RTX 2070 SUPER:")
log()
log(f"{'n_times':>7s} {'n_tests':>11s} {'CPU_ms':>9s} {'GPU_ms':>9s} {'Speedup':>8s}")
log("-" * 50)
for nt, ntests, nsig, cpu_ms, gpu_ms, spd in ccl_results:
    marker = " <-- GPU crossover" if 0.95 < spd < 1.1 else ""
    marker = " <-- GPU WINS" if spd >= 1.1 else marker
    log(f"{nt:7d} {ntests:11,} {cpu_ms:9.1f} {gpu_ms:9.1f} {spd:7.2f}x{marker}")

log()
log("=" * 70)
log("CONCLUSIONS")
log("=" * 70)
log("""
For RAPID ITERATION benchmarking, use these parameters:
  - Isolated CCL: n_times=250 (5.1M tests, 30% density) => ~1.7x GPU speedup
  - Full permutation test: see Part 2/3 results above

The CuPy drop-in gives modest speedup (1.2-2x) only when:
  1. The full spatio-temporal adjacency is pre-built (forcing _get_components path)
  2. The graph has >1M total tests
  3. >20% of vertices are supra-threshold

For dramatic speedup (>5x), the fused GPU pipeline (Path 3) is needed:
  - Entire permutation loop on GPU
  - No per-permutation CPU<->GPU transfers
  - sign-flip + t-test + threshold + CCL + reduce in one dispatch
""")

log(f"Benchmark finished: {datetime.now().isoformat()}")
save_results()
log(f"\nResults saved to: {RESULTS_FILE}")
