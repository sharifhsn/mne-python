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

## Empirical Results (2026-03-08, RTX 2070 SUPER)

### Critical finding: wrong bottleneck target

The original analysis identified `_get_components()` (SciPy sparse CCL) as the
bottleneck. **This is incorrect.** In the standard spatio-temporal case:

1. `_setup_adjacency()` detects that the spatial adjacency (20,484 × 20,484) is
   smaller than `n_tests` (307,260) and converts it to a **neighbor list**.
2. This routes clustering to `_get_clusters_st()` → `_get_clusters_spatial()`
   (Numba JIT BFS on neighbor lists), **NOT** `_get_components()`.
3. `_get_components()` is called **0 times** in the standard benchmark.

### Profiled time breakdown (p<0.05, 128 perms)

| Component | Time | % |
|-----------|------|---|
| `_get_clusters_st` (Numba BFS) | 21.4s | 91.9% |
| `_find_clusters` overhead | 1.3s | 5.4% |
| stat_fun + sign-flip | 0.6s | 2.7% |
| **Total** | **23.3s** | |

The bottleneck claim (~97% in CCL) is correct — but the CCL implementation
is Numba BFS on neighbor lists, not SciPy sparse graph CCL.

### CuPy drop-in results: GPU is SLOWER

| Configuration | CPU | GPU | Speedup |
|---------------|-----|-----|---------|
| Full ST adjacency, p<0.05 | 23.5s | 61.2s | **0.38x** |
| Isolated CCL, 20K vertices | 1.0ms | 14.8ms | **0.07x** |
| Isolated CCL, 500K vertices | 24.6ms | 40.7ms | **0.60x** |
| Isolated CCL, 1M vertices | 67ms | 63ms | **1.07x** (crossover) |
| Isolated CCL, 2M vertices | 217ms | 117ms | **1.85x** |
| Isolated CCL, 5M vertices | 694ms | 302ms | **2.30x** |

GPU only wins at >1M vertices per call. MNE's typical graph has 307K nodes
with only 1K-10K supra-threshold per permutation.

### Why CuPy fails here

1. **Kernel launch overhead**: ~15ms per `pylibcugraph.weakly_connected_components`
   call, while CPU completes in <1ms for small subgraphs.
2. **Transfer overhead**: Building COO matrix on CPU, transferring to GPU, and
   copying labels back adds ~5ms per call.
3. **Many small calls**: 128+ calls per test run (2 tails × n_permutations),
   each on a different subgraph. GPU can't batch these.

### Path 1 verdict: NOT VIABLE

The CuPy drop-in approach cannot provide speedup for MNE's permutation cluster
tests. The per-call overhead (~20ms) exceeds the CPU compute time (<1ms for
typical subgraphs).

### Revised plan

Path 1 is eliminated. The viable approaches are:

1. **Path 3 (fused GPU pipeline)**: Keep the entire permutation loop on GPU —
   sign-flip, t-test, threshold, CCL, reduce — in a single dispatch. This
   amortizes kernel launch overhead across all permutations. Expected: 10-50x.

2. **Alternative: Numba CUDA**: Replace the Numba CPU JIT in `_get_clusters_spatial`
   with Numba CUDA kernels. Same Python ecosystem, no Rust/PyO3 needed.
   Lower effort than Path 3 but less potential speedup.

3. **Alternative: batched GPU CCL**: Run CCL for multiple permutations in a single
   GPU dispatch (e.g., batch 64 subgraphs into one kernel). Requires custom
   CUDA kernel (not available in CuPy/pylibcugraph).

## Recommended Plan

1. ~~**Path 1**: CuPy drop-in~~ — **ELIMINATED** (GPU slower, see results above)
2. **Next**: Prototype fused GPU pipeline (Path 3) or batched Numba CUDA
3. **Key insight**: Must avoid per-permutation CPU↔GPU round-trips
4. **Benchmark**: `benchmark_head_to_head.py` has the full comparison

## References

- [GPU CCL: Jaiganesh & Burtscher (HPDC 2018)](https://userweb.cs.txstate.edu/~mb92/papers/hpdc18.pdf)
- [CuPy connected_components](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.csgraph.connected_components.html)
- [wgpu — Portable GPU API for Rust](https://wgpu.rs/)
- [MNE permutation_cluster_test docs](https://mne.tools/stable/generated/mne.stats.permutation_cluster_test.html)
- [MNE tutorial: source-space cluster test](https://mne.tools/stable/auto_tutorials/stats-source-space/20_cluster_1samp_spatiotemporal.html)
