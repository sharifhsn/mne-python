#!/usr/bin/env python3
"""
Proof-of-concept: CuPy GPU backend for cluster-level connected components.

This script monkey-patches MNE's `_get_components` to use CuPy's GPU
connected_components, then runs the same benchmark as the CPU baseline.

Requirements (Linux with NVIDIA GPU):
    uv pip install cupy-cuda12x  # or cupy-cuda11x for older CUDA

Usage:
    python gpu_accel/patch_cupy_poc.py
"""

import sys
import time

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse
    from cupyx.scipy.sparse.csgraph import connected_components as gpu_cc
    HAS_CUPY = True
    print(f"CuPy {cp.__version__} detected, CUDA device: {cp.cuda.Device().name}")
except ImportError:
    HAS_CUPY = False
    print("CuPy not available. Install with: uv pip install cupy-cuda12x")
    sys.exit(1)

from scipy import sparse, stats

import mne
from mne.stats import cluster_level as cl


# ---------- Monkey-patch _get_components with GPU backend ----------

_original_get_components = cl._get_components


def _gpu_get_components(x_in, adjacency, return_list=True):
    """GPU-accelerated connected components via CuPy."""
    if adjacency is False:
        components = np.arange(len(x_in))
    else:
        # Build subgraph of significant vertices (same logic as original)
        mask = np.logical_and(x_in[adjacency.row], x_in[adjacency.col])
        data = adjacency.data[mask]
        row = adjacency.row[mask]
        col = adjacency.col[mask]
        shape = adjacency.shape
        idx = np.where(x_in)[0]
        row = np.concatenate((row, idx))
        col = np.concatenate((col, idx))
        data = np.concatenate((data, np.ones(len(idx), dtype=data.dtype)))

        # Transfer to GPU and run connected components
        adj_gpu = cp_sparse.coo_matrix(
            (cp.asarray(data), (cp.asarray(row), cp.asarray(col))),
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


# Apply the patch
cl._get_components = _gpu_get_components
print("Patched _get_components with GPU backend.\n")


# ---------- Run benchmark ----------

from benchmark_cluster_cpu import benchmark, load_source_data

print("Loading source-space data from MNE sample dataset...")
X, adjacency, n_subjects = load_source_data()
print("Data loaded.\n")

# Also pre-transfer adjacency to GPU to measure warm start
adj_coo = adjacency.tocoo()
print(f"Adjacency transferred: {adj_coo.nnz} nonzeros, "
      f"{adj_coo.data.nbytes / 1024:.0f} KB\n")

print("=" * 70)
print("GPU (CuPy) BENCHMARK")
print("=" * 70)

# Note: n_jobs=1 because GPU handles parallelism internally.
# Joblib parallelism + GPU would cause contention.
results = benchmark(
    X, adjacency, n_subjects,
    n_perms_list=[64, 256, 1024],
    n_jobs_list=[1],
)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for r in results:
    rate = r["n_perms"] / r["elapsed"]
    print(
        f"  {r['n_perms']:5d} perms: "
        f"{r['elapsed']:6.1f}s ({rate:.1f} perms/s)"
    )
