# GPU-Accelerated Permutation Cluster Tests for MNE-Python

## Problem

Cluster-based permutation tests (`mne.stats.spatio_temporal_cluster_1samp_test`)
are the #1 computational bottleneck for MNE researchers. For source-space
analyses (fsaverage ico-5, ~20K vertices), a typical run with 5,000 permutations
takes **10-20 minutes** on a modern CPU.

~97% of this time is spent in **connected-component labeling** on a sparse
adjacency graph — not in the t-test computation or data shuffling.

Nobody has attempted GPU acceleration of this routine in any neuroimaging
package. This is a genuine gap.

## Why Connected Components on GPU Works

Connected-component labeling (CCL) on sparse graphs is a solved GPU problem:

- **Jaiganesh & Burtscher (HPDC 2018)** — GPU union-find with atomic CAS and
  path compression. 100x speedup on large graphs.
- **SciPy's `connected_components()`** runs BFS on CPU (single-threaded).
- **CuPy already ships `cupyx.scipy.sparse.csgraph.connected_components()`** —
  a GPU drop-in for NVIDIA hardware.

The algorithm requires only basic GPU features: atomics (compare-and-swap),
storage buffers, and workgroup dispatch. These are supported by every GPU
since 2012.

## Architecture: Three Paths

### Path 1: CuPy Drop-in (proof of concept, NVIDIA-only)

Patch `mne/stats/cluster_level.py::_get_components()` to use CuPy:
```python
try:
    import cupy as cp
    from cupyx.scipy.sparse.csgraph import connected_components as gpu_cc
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
```

**Effort**: Days. **Expected**: 10-50x on clustering step. **Limitation**: NVIDIA only.

### Path 2: wgpu + Rust (cross-platform)

Write a union-find compute shader in WGSL, dispatch via `wgpu` from Rust,
expose to Python via PyO3. Works on NVIDIA (Vulkan), AMD (Vulkan),
Apple Silicon (Metal), Intel (Vulkan).

**Effort**: 2-4 weeks. **Expected**: Similar to CuPy.

### Path 3: Fused GPU pipeline (maximum speedup)

Keep entire permutation loop on GPU — shuffle, t-test, threshold, CCL,
reduce — in a single Rust binary. Zero CPU-GPU transfer per permutation.

**Effort**: 2-3 months. **Expected**: 50-200x total (17 min → 5-20 sec).

## Hardware Requirements

| API | Minimum GPU | Year |
|-----|------------|------|
| Vulkan 1.0 | GTX 600 series | 2012 |
| Metal | Apple A7 / M1 | 2013/2020 |
| DirectX 12 | Most GPUs from 2015+ | 2015 |
| wgpu (auto) | All of the above | — |

No tensor cores, ray tracing, or modern GPU features required.

## MNE Repo Prior Art

| PR/Issue | What | Outcome |
|----------|------|---------|
| #13002 | CUDA zero-copy for resampling | Closed (dep issues) |
| #12609 | TFCE memory/speed optimization | **Merged** |
| #7784 | Minor permutation test speedup | **Merged** (2-5%) |
| #8095 | Numba for summarize_clusters_stc | **Merged** (35%) |
| #5439 | Use CuPy for linalg | Open (stalled, modest gains) |
| #13175 | Add uv environment support | Open (maintainers already use uv) |

Maintainers (`larsoner`, `mscheltienne`) are receptive to performance work.
Nobody has tried GPU for permutation tests specifically.

## Benchmark Dataset

The MNE `sample` dataset (~1.5 GB, auto-downloads) provides a ready-made
source-space benchmark. See `benchmark_cluster_cpu.py` in this directory.

Typical dimensions:
- **n_vertices**: ~20,484 (fsaverage ico-5)
- **n_times**: ~15 timepoints
- **n_tests**: ~307,260
- **adjacency**: 20,484 × 20,484 sparse (~123K nonzero entries, ~1 MB on GPU)

## Recommended Plan

1. **Today**: Run `benchmark_cluster_cpu.py` to establish CPU baseline
2. **This week**: Patch `_get_components` with CuPy backend, benchmark same data
3. **If speedup confirmed**: Build wgpu+Rust cross-platform version
4. **Then**: Open PR to MNE-Python with benchmarks

## References

- [GPU CCL: Jaiganesh & Burtscher (HPDC 2018)](https://userweb.cs.txstate.edu/~mb92/papers/hpdc18.pdf)
- [CuPy connected_components](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.csgraph.connected_components.html)
- [wgpu — Portable GPU API for Rust](https://wgpu.rs/)
- [MNE permutation_cluster_test docs](https://mne.tools/stable/generated/mne.stats.permutation_cluster_test.html)
- [MNE tutorial: source-space cluster test](https://mne.tools/stable/auto_tutorials/stats-source-space/20_cluster_1samp_spatiotemporal.html)
