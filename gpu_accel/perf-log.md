# Performance Log

## 2026-03-08: Vectorize cluster-list-building loop in `_get_components`

### What changed

Replaced the pure-Python for-loop in `_get_components` (lines 305-313 of
`mne/stats/cluster_level.py`) that converts component labels into cluster lists
with vectorized NumPy operations (`bincount` + `argsort` + `split`).

Also added a fast path for `adjacency is False` that skips component-label
machinery entirely and uses `np.where` directly.

Same fix applied to GPU copies in `gpu_accel/patch_cupy_poc.py` and
`gpu_accel/benchmark_head_to_head.py`.

### Why

Profiling showed 92.3% of GPU `_get_components` time (3,279ms of 3,554ms) was
spent in this loop. It iterates over every vertex (2M+ in spatio-temporal
workloads), calling `list.append()` and doing Python-level bookkeeping per
element.

### Old code

```python
comp_list = [list() for i in range(start, stop + 1, 1)]
mask = np.zeros(len(comp_list), dtype=bool)
for ii, comp in enumerate(components):
    comp_list[comp].append(ii)
    mask[comp] += x_in[ii]
clusters = [np.array(k) for k, m in zip(comp_list, mask) if m]
```

### New code

```python
has_sig = np.bincount(
    components, weights=x_in.astype(bool).astype(np.float64)
) > 0
order = np.argsort(components, kind="stable")
counts = np.bincount(components)
splits = np.cumsum(counts[:-1])
all_clusters = np.split(order, splits)
clusters = [c for c, m in zip(all_clusters, has_sig) if m]
```

### Benchmark results (43/43 parity checks passed, 0 regressions)

#### Varying supra-threshold density (500K vertices, sparse adjacency)

| Density | Old (ms) | New (ms) | Speedup |
|---------|----------|----------|---------|
| 1%      | 580      | 340      | 1.7x    |
| 5%      | 562      | 346      | 1.6x    |
| 10%     | 555      | 352      | 1.6x    |
| 30%     | 546      | 310      | 1.8x    |
| 50%     | 631      | 346      | 1.8x    |
| 70%     | 723      | 387      | 1.9x    |
| 95%     | 882      | 490      | 1.8x    |

#### Varying graph size (30% supra-threshold)

| Vertices    | Old (ms) | New (ms) | Speedup |
|-------------|----------|----------|---------|
| 1,000       | 1.1      | 0.7      | 1.6x    |
| 20,000      | 19.7     | 11.2     | 1.8x    |
| 100,000     | 108      | 58       | 1.9x    |
| 500,000     | 558      | 304      | 1.8x    |
| 1,000,000   | 1,212    | 661      | 1.8x    |
| 2,000,000   | 2,582    | 1,502    | 1.7x    |
| 3,000,000   | 3,971    | 2,168    | 1.8x    |

#### `adjacency=False` (pure loop, no CCL)

| Vertices    | Old (ms) | New (ms) | Speedup |
|-------------|----------|----------|---------|
| 1,000       | 1.0      | 0.04     | 26.9x   |
| 100,000     | 110      | 4.0      | 27.4x   |
| 1,000,000   | 1,307    | 55       | 23.7x   |
| 2,000,000   | 2,486    | 101      | 24.6x   |

#### Isolated loop-only timing (CCL cost excluded)

| Vertices    | Old (ms) | New (ms) | Speedup |
|-------------|----------|----------|---------|
| 10,000      | 8.7      | 4.7      | 1.9x    |
| 100,000     | 96       | 47       | 2.0x    |
| 500,000     | 489      | 241      | 2.0x    |
| 1,000,000   | 1,030    | 503      | 2.0x    |
| 2,000,000   | 2,109    | 1,012    | 2.1x    |

#### Real MNE source-space (fsaverage ico-5, 20,484 vertices)

| Density | Old (ms) | New (ms) | Speedup |
|---------|----------|----------|---------|
| 5%      | 21.0     | 15.0     | 1.4x    |
| 10%     | 20.8     | 14.0     | 1.5x    |
| 30%     | 21.0     | 12.3     | 1.7x    |
| 50%     | 20.2     | 9.7      | 2.1x    |

#### Spatio-temporal (fsaverage x timepoints, 10% supra)

| Configuration          | Old (ms) | New (ms) | Speedup |
|------------------------|----------|----------|---------|
| 20,484 x 5 = 102K     | 149      | 72       | 2.1x    |
| 20,484 x 15 = 307K    | 448      | 227      | 2.0x    |
| 20,484 x 25 = 512K    | 710      | 409      | 1.7x    |

### Interpretation

- The vectorized loop replacement is a consistent **2x** improvement in isolation.
- End-to-end speedup is **1.5-2x** when sparse CCL (`connected_components`) is
  included, since CCL accounts for ~50% of total time and is unchanged.
- The `adjacency=False` path sees **24-27x** improvement because the loop was
  100% of the cost and is now replaced by a single `np.where`.
- The remaining bottleneck is `scipy.sparse.csgraph.connected_components` itself
  AND `np.split` creating thousands of tiny array objects (67-90% of remaining time).

---

## 2026-03-08: Reindex to active-only vertices in `_get_components`

### What changed

Instead of running `connected_components` on the full N-vertex graph (where most
vertices are inactive singletons), build a compact graph containing only supra-
threshold vertices. This makes CCL faster (smaller graph) and eliminates the
`np.split` bottleneck (far fewer components to partition).

Also eliminates the `has_sig` filtering step entirely — since the compact graph
only contains active vertices, every component is a valid cluster by construction.

### Why

Profiling the previous optimization (v1) revealed `np.split` was 67-90% of
`_get_components` time, creating ~15K-20K tiny numpy arrays. The compact-graph
approach reduces this to ~800-2K components at typical supra-threshold densities.

### Code (return_list=True path)

```python
idx = np.where(x_in)[0]
n_active = len(idx)
if n_active == 0:
    return []
# Map global → local indices
global_to_local = np.empty(adjacency.shape[0], dtype=np.intp)
global_to_local[idx] = np.arange(n_active)
# Keep only edges between active vertices, remap to local
edge_mask = np.logical_and(x_in[adjacency.row], x_in[adjacency.col])
row = global_to_local[adjacency.row[edge_mask]]
col = global_to_local[adjacency.col[edge_mask]]
# Self-loops for isolated active vertices
self_idx = np.arange(n_active)
row = np.concatenate((row, self_idx))
col = np.concatenate((col, self_idx))
data = np.ones(len(row), dtype=np.float64)
small_adj = sparse.coo_array((data, (row, col)), shape=(n_active, n_active))
_, components = connected_components(small_adj)
# Group and map back to global indices
order = np.argsort(components, kind="stable")
counts = np.bincount(components)
splits = np.cumsum(counts[:-1])
global_order = idx[order]
return list(np.split(global_order, splits))
```

### Benchmark results (34/34 parity checks passed, 0 regressions)

Three-way comparison: old (Python loop) → v1 (vectorized full-graph) → v2 (reindex).

#### End-to-end permutation test (64 perms, fsaverage ico-5, 20K vertices)

| Version         | Time   | vs old | Clusters | Significant |
|-----------------|--------|--------|----------|-------------|
| old (loop)      | 2.09s  | 1x     | 2,225    | 183         |
| v1 (vectorized) | 1.05s  | 2.0x   | 2,225    | 183         |
| **v2 (reindex)**| **0.20s** | **10.7x** | 2,225 | 183      |

#### Varying supra-threshold density (500K vertices, sparse adjacency)

| Density | Old (ms) | v1 (ms) | v2 (ms) | v2/old | v2/v1 |
|---------|----------|---------|---------|--------|-------|
| 1%      | 664      | 362     | 14.8    | 45.0x  | 24.5x |
| 5%      | 659      | 340     | 25.5    | 25.8x  | 13.3x |
| 10%     | 691      | 341     | 35.6    | 19.4x  | 9.6x  |
| 30%     | 651      | 306     | 54.5    | 11.9x  | 5.6x  |
| 50%     | 669      | 310     | 127     | 5.3x   | 2.4x  |
| 70%     | 779      | 392     | 260     | 3.0x   | 1.5x  |
| 95%     | 882      | 493     | 519     | 1.7x   | 0.9x  |

#### Varying graph size (30% supra-threshold)

| Vertices    | Old (ms) | v1 (ms) | v2 (ms) | v2/old |
|-------------|----------|---------|---------|--------|
| 1,000       | 1.2      | 0.7     | 0.2     | 5.7x   |
| 5,000       | 5.4      | 3.2     | 0.6     | 9.1x   |
| 20,000      | 20.5     | 12.1    | 1.8     | 11.3x  |
| 100,000     | 142      | 64      | 9.5     | 14.9x  |
| 500,000     | 669      | 299     | 55.5    | 12.1x  |
| 1,000,000   | 1,357    | 698     | 160     | 8.5x   |
| 2,000,000   | 2,710    | 1,408   | 350     | 7.7x   |

#### Real fsaverage ico-5 (20,484 vertices)

| Density | Old (ms) | v1 (ms) | v2 (ms) | v2/old | v2/v1 |
|---------|----------|---------|---------|--------|-------|
| 5%      | 19.4     | 13.6    | 0.9     | 20.5x  | 14.4x |
| 10%     | 19.5     | 13.7    | 1.4     | 13.5x  | 9.4x  |
| 20%     | 20.0     | 13.0    | 2.1     | 9.7x   | 6.3x  |
| 30%     | 20.1     | 12.3    | 2.3     | 8.6x   | 5.3x  |
| 50%     | 20.2     | 9.7     | 2.2     | 9.1x   | 4.3x  |

#### Spatio-temporal (fsaverage × timepoints, 10% supra)

| Configuration          | Old (ms) | v1 (ms) | v2 (ms) | v2/old | v2/v1 |
|------------------------|----------|---------|---------|--------|-------|
| 20,484 × 5 = 102K     | 133      | 70      | 6.7     | 19.8x  | 10.4x |
| 20,484 × 15 = 307K    | 439      | 216     | 24.0    | 18.3x  | 9.0x  |
| 20,484 × 25 = 512K    | 749      | 405     | 44.1    | 17.0x  | 9.2x  |

#### Edge cases

| Scenario                   | Old (ms) | v2 (ms) | v2/old  |
|----------------------------|----------|---------|---------|
| All active, 50K            | 61.1     | 19.1    | 3.2x    |
| Single active, 50K         | 81.0     | 0.8     | 97.5x   |
| No active, 50K             | 60.2     | 0.0     | instant |
| Sparse adj (deg=1), 50K    | 56.0     | 8.4     | 6.6x    |
| Dense adj (deg=50), 5K     | 9.7      | 4.5     | 2.1x    |
| Integer x_in (label.py)    | 16.2     | 3.1     | 5.2x    |
| Integer x_in w/ vertex 0   | 1.7      | 0.4     | 4.1x    |

### Interpretation

- At realistic supra-threshold densities (5-30%), v2 is **9-20x faster** than
  the original and **5-14x faster** than v1.
- The gain comes from two sources: smaller graph for `connected_components`
  (faster CCL) and far fewer components for `np.split` (less object creation).
- At 95% supra-threshold (nearly all vertices active), the compact graph is
  almost as large as the full graph, so v2 ≈ v1 (0.9x). This is acceptable
  since 95% density is unrealistic in practice.
- End-to-end on 64 permutations: **10.7x faster** than original (2.09s → 0.20s).
- All 34 parity checks passed; all 55 unit tests passed.

---

## 2026-03-08: Vectorize cluster sums with `np.add.reduceat`

### What changed

Replaced the per-cluster `_masked_sum` list comprehension in `_find_clusters_1dir`
with a single `np.add.reduceat` call over the concatenated cluster indices.

Old code (line 581-584):
```python
if t_power == 1:
    sums = [_masked_sum(x, c) for c in clusters]
else:
    sums = [_masked_sum_power(x, c, t_power) for c in clusters]
```

New code:
```python
all_idx = np.concatenate(clusters)
lengths = np.array([len(c) for c in clusters])
offsets = np.empty(len(clusters), dtype=np.intp)
offsets[0] = 0
np.cumsum(lengths[:-1], out=offsets[1:])
if t_power == 1:
    sums = np.add.reduceat(x[all_idx], offsets)
else:
    vals = np.sign(x[all_idx]) * np.abs(x[all_idx]) ** t_power
    sums = np.add.reduceat(vals, offsets)
```

### Why

With `_get_components` already optimized (reindex), cluster sums were the next
per-permutation bottleneck: 0.43ms/call making ~1,500 Python calls to
`np.sum(x[c])`. In isolation, `np.bincount` is 76x faster. The `reduceat`
approach doesn't require changing `_get_components`'s return type and also
benefits the `_get_clusters_st` (Numba neighbor-list) code path.

### Benchmark results (14/14 parity checks passed, 0 regressions)

#### Per-call: _find_clusters with reduceat vs _masked_sum loop (fsaverage)

| Density | Clusters | Old (ms) | New (ms) | Speedup |
|---------|----------|----------|----------|---------|
| 1%      | 194      | 0.55     | 0.53     | 1.0x    |
| 5%      | 875      | 1.30     | 1.18     | 1.1x    |
| 10%     | 1,432    | 1.98     | 1.69     | 1.2x    |
| 20%     | 2,011    | 2.63     | 2.38     | 1.1x    |
| 30%     | 1,705    | 2.74     | 2.52     | 1.1x    |

Per-call improvement is modest (~1.1x) because `_get_components` still
dominates. The sums savings accumulate over many permutations.

#### End-to-end (128 perms, fsaverage ico-5)

| Version                      | Time   | vs original |
|------------------------------|--------|-------------|
| Original (v0: loop + sums)   | 2.09s  | 1x          |
| v2 (reindex, old sums)       | 0.401s | 5.2x        |
| **v3 (reindex + reduceat)**  | **0.327s** | **6.4x** |

### Interpretation

- The reduceat optimization saves ~0.4ms × 128 perms = ~50ms, bringing the
  end-to-end test from 0.401s to 0.327s (18% faster).
- Cumulative speedup from all three optimizations: **6.4x** vs original.
- Per-permutation breakdown is now: `_get_components` ~1.0ms (65%),
  stat function ~0.4ms (26%), sums+overhead ~0.15ms (9%).
- The remaining per-permutation bottleneck is `connected_components` and
  `np.argsort` inside `_get_components`.

---

## 2026-03-08: Skip `np.split` in permutation loop (`_sums_only` fast path)

### What changed

Added a `_sums_only` parameter that threads through `_find_clusters` →
`_find_clusters_1dir_parts` → `_find_clusters_1dir`. When `True`, the sparse
adjacency path calls `_get_components(return_labels=True)` which returns raw
`(idx, components)` instead of building cluster lists (skips `argsort`/`split`).
Sums are computed directly via `np.bincount(components, weights=x[idx])`.

Also added `return_labels=False` parameter to `_get_components`. When `True`,
returns `(idx, components)` tuple — the active vertex indices and their component
labels — skipping `argsort`, `bincount`, `cumsum`, and `np.split` entirely.

Both `_do_1samp_permutations` and `_do_permutations` now pass `_sums_only=True`
since they only use `out[1]` (sums), never the cluster list.

TFCE is protected: `_sums_only` is forced `False` when `tfce=True` since
TFCE scoring needs cluster lists internally.

### Why

After v3, the per-permutation breakdown was: `_get_components` ~1.0ms (65%),
stat function ~0.4ms (26%), sums ~0.15ms (9%). Within `_get_components`,
`np.split` creating hundreds of tiny numpy arrays was 55-62% of the time,
and the resulting cluster lists were immediately re-concatenated in the
reduceat sums code. Profiling showed `np.bincount` with weights (no split) is
106x faster than split→concatenate→reduceat for the sums computation.

### Benchmark results (23/23 parity checks passed, 0 regressions)

#### Per-call: _find_clusters full vs _sums_only (fsaverage, tail=1)

| Density | Clusters | Full (ms) | Sums-only (ms) | Speedup |
|---------|----------|-----------|----------------|---------|
| 1%      | 194      | 0.57      | 0.38           | 1.5x    |
| 5%      | 875      | 1.24      | 0.46           | 2.7x    |
| 10%     | 1,432    | 1.85      | 0.58           | 3.2x    |
| 20%     | 2,011    | 2.70      | 0.89           | 3.0x    |
| 30%     | 1,705    | 2.88      | 1.24           | 2.3x    |
| 50%     | 409      | 3.00      | 2.04           | 1.5x    |

#### Two-tailed (tail=0, fsaverage)

| Clusters | Full (ms) | Sums-only (ms) | Speedup |
|----------|-----------|----------------|---------|
| 3,733    | 4.76      | 1.49           | 3.2x    |

#### adjacency=False

| Clusters | Full (ms) | Sums-only (ms) | Speedup |
|----------|-----------|----------------|---------|
| 2,115    | 0.67      | 0.04           | 19.1x   |

#### _get_components: return_labels vs return_list (fsaverage)

| Density | Clusters | return_list (ms) | return_labels (ms) | Speedup |
|---------|----------|------------------|--------------------|---------|
| 5%      | 852      | 1.01             | 0.40               | 2.5x    |
| 10%     | 1,459    | 1.58             | 0.53               | 3.0x    |
| 30%     | 1,735    | 2.55             | 1.21               | 2.1x    |

#### Varying graph size (30% supra)

| Vertices    | Full (ms) | Sums-only (ms) | Speedup |
|-------------|-----------|----------------|---------|
| 1,000       | 0.30      | 0.23           | 1.3x    |
| 10,000      | 1.07      | 0.94           | 1.1x    |
| 50,000      | 4.63      | 4.16           | 1.1x    |
| 100,000     | 10.1      | 8.8            | 1.1x    |
| 500,000     | 78.7      | 76.2           | 1.0x    |

#### End-to-end (128 perms, fsaverage ico-5)

| Version                      | Time   | vs original |
|------------------------------|--------|-------------|
| Original (v0: loop + sums)   | 2.09s  | 1x          |
| v3 (reindex + reduceat)      | 0.327s | 6.4x        |
| **v4 (+ _sums_only)**        | **0.300s** | **7.0x** |

### Interpretation

- Per-call speedup is **1.5-3.2x** on fsaverage because `return_labels` skips
  `argsort` + `np.split` (the expensive groupby step) and `bincount` replaces
  `concatenate` + `reduceat`.
- End-to-end improvement is modest (0.327s → 0.300s, ~9%) because
  `connected_components` itself now dominates (~80% of per-permutation time).
- At large graph sizes (500K+), `connected_components` is nearly 100% of the
  cost, so the split-avoidance savings are proportionally tiny.
- `adjacency=False` sees **19x** improvement because cluster-list construction
  was the dominant cost and is now eliminated.
- Cumulative speedup: **7.0x** vs original (2.09s → 0.300s).
- The remaining bottleneck is `scipy.sparse.csgraph.connected_components` itself
  — a single-threaded BFS on the compact graph. This is the hard limit of what
  can be achieved without replacing the CCL algorithm (GPU, parallel BFS, etc.).

---

## 2026-03-08: Numba union-find + precomputed t-test (v5)

### What changed

Two optimizations applied together:

**1. Numba union-find replaces scipy `connected_components`** in the permutation
loop (`return_labels=True` path of `_get_components`).

Added `_fused_ccl` — a Numba JIT function that iterates the full adjacency edge
list once, checking `x_in[r] and x_in[c]` per edge and unioning active vertices.
This replaces the entire numpy edge-filtering → sparse matrix construction →
scipy CSC conversion → BFS pipeline with a single tight Numba loop.

Falls back to the scipy path when Numba is not available (same `has_numba` guard
used by existing MNE JIT functions).

**2. Precomputed t-test for sign-flip permutations** in `_do_1samp_permutations`.

Exploits the algebraic identity that for sign-flips s (±1), s²=1, so:
- `sum(X² * s²) = sum(X²)` — precomputable once before the permutation loop
- `mean_s = signs @ X / n` — a matrix-vector multiply (20 × 20K), much cheaper
  than two full-array multiplies (20 × 20K each) + `np.var`
- `var_s = (sum_sq - n * mean_s²) / (n-1)` — element-wise, no data movement

Activates only when `stat_fun is ttest_1samp_no_p` (identity check — custom
functions or partial applications correctly fall back to the standard path).

### Why

After v4, the per-permutation breakdown was:
- `connected_components` + sparse build: ~80% (~0.45ms)
- `ttest_1samp_no_p` + sign flip: ~15% (~0.5ms with buffer_size)
- bincount sums: ~5%

The scipy `connected_components` path was bottlenecked by:
1. Sparse matrix construction (`coo_array` with data/row/col arrays)
2. Internal CSC conversion (sorting + index building)
3. BFS traversal

The Numba union-find eliminates all three — just iterates edges and unions.

The ttest optimization was hidden because the default `buffer_size=1000` was
preventing activation. Once fixed, it saves ~0.4ms per permutation by avoiding
two full-array sign-flip multiplies and the `np.var` call.

### Benchmark results (19/19 parity checks passed, 0 regressions)

#### _get_components return_labels: scipy vs Numba UF (fsaverage)

| Density | Components | scipy (ms) | UF (ms) | Speedup |
|---------|------------|-----------|---------|---------|
| 1%      | 194        | 0.34      | 0.09    | 4.0x    |
| 5%      | 875        | 0.42      | 0.11    | 4.0x    |
| 10%     | 1,432      | 0.54      | 0.14    | 4.0x    |
| 20%     | 2,011      | 0.84      | 0.21    | 3.9x    |
| 30%     | 1,705      | 1.21      | 0.32    | 3.8x    |
| 50%     | 409        | 1.96      | 0.56    | 3.5x    |

#### _find_clusters sums_only: full pipeline (fsaverage)

| Density | Clusters | v4 scipy (ms) | v5 UF (ms) | Speedup |
|---------|----------|---------------|-----------|---------|
| 5%      | 859      | 1.20          | 0.13      | 9.3x    |
| 10%     | 1,500    | 1.88          | 0.17      | 11.4x   |
| 30%     | 1,741    | 2.92          | 0.36      | 8.0x    |

#### Varying graph size (30% supra)

| Vertices    | scipy (ms) | UF (ms) | Speedup |
|-------------|-----------|---------|---------|
| 1,000       | 0.18      | 0.02    | 11.4x   |
| 5,000       | 0.50      | 0.11    | 4.5x    |
| 20,000      | 1.69      | 0.53    | 3.2x    |
| 100,000     | 12.4      | 3.2     | 3.8x    |
| 500,000     | 68.5      | 19.6    | 3.5x    |

#### Precomputed ttest vs standard path

| Path                     | 128 perms | Savings |
|--------------------------|-----------|---------|
| Default (precomputed)    | 0.093s    | —       |
| Custom stat_fun (fallback)| 0.229s   | 0.136s  |

#### End-to-end (128 perms, fsaverage ico-5)

| Version                      | Time    | vs original |
|------------------------------|---------|-------------|
| Original (v0: loop + sums)   | 2.09s   | 1x          |
| v3 (reindex + reduceat)      | 0.327s  | 6.4x        |
| v4 (+ _sums_only)            | 0.300s  | 7.0x        |
| **v5 (UF + precomp ttest)**  | **0.093s** | **22.4x** |

### Interpretation

- Numba union-find is **3.5-11x faster** than scipy `connected_components` for
  the sums-only permutation path, eliminating sparse matrix construction overhead.
- Precomputed ttest saves **~1ms/perm** (0.136s over 128 perms) by exploiting
  the identity s²=1 for sign-flip permutations.
- Combined: **22.4x cumulative speedup** vs original (2.09s → 0.093s).
- The 0.093s breaks down as: ~0.05s overhead (initial clustering, mask
  conversion, p-values) + ~0.04s permutation loop (128 × ~0.3ms/perm).
- At 0.3ms/perm, remaining costs are: Numba UF edge iteration (~50%), threshold
  comparison + np.where (~20%), precomputed ttest (~20%), bincount sums (~10%).

---

## 2026-03-08: Reduce per-perm overhead (v6)

### What changed

Three micro-optimizations to reduce per-permutation allocation overhead:

**1. Skip `include` array allocation in `_find_clusters`** when `include is None`.
Previously, `np.ones(x.shape, dtype=bool)` was allocated every call (even in the
permutation loop) just to pass as a mask that includes everything. Now uses a
`_has_include` flag to branch: when `False`, thresholding is done directly
(`x > thresh`) instead of `np.logical_and(x > thresh, include)`.

**2. Pre-allocate work arrays in `_do_1samp_permutations`** for the fast ttest
path. `_mean_s`, `_denom_sq`, and `_t_buf` (each 20K doubles) are allocated once
before the loop. All intermediate operations use in-place NumPy
(`np.dot(..., out=)`, `np.multiply(..., out=)`, `np.maximum(..., out=)`,
`np.sqrt(..., out=)`, `np.divide(..., out=)`).

**3. Simplified signs computation**: `signs_1d = 2.0 * order - 1.0` instead of
`signs = 2 * order[:, None].astype(int) - 1` followed by `.ravel().astype(float64)`.
Avoids reshape, copy, and dtype conversion.

### Why

Profiling with 1024 perms showed:
- `_find_clusters` wrapper: ~36us/call from `include` allocation (37ms total)
- Precomputed ttest: 5-6 temporary 20K arrays per perm (~15us/perm in GC pressure)
- Signs computation: extra reshape/ravel/astype for an already-1D boolean array

### Benchmark results (4/4 parity checks passed, all 55 unit tests passed)

#### End-to-end (fsaverage ico-5, 20,484 vertices)

| Perms | Time    | Per-perm |
|-------|---------|----------|
| 64    | 0.044s  | —        |
| 128   | 0.077s  | —        |
| 256   | 0.178s  | —        |
| 512   | 0.280s  | —        |
| 1024  | 0.582s  | —        |
| 2048  | 1.153s  | —        |

Linear regression: overhead = 2.9ms, per_perm = 0.56ms

#### Cumulative progress

| Version                      | 128 perms | vs original |
|------------------------------|-----------|-------------|
| Original (v0: loop + sums)   | 2.09s     | 1x          |
| v3 (reindex + reduceat)      | 0.327s    | 6.4x        |
| v4 (+ _sums_only)            | 0.300s    | 7.0x        |
| v5 (UF + precomp ttest)      | 0.093s    | 22.4x       |
| **v6 (reduced overhead)**    | **0.077s** | **27.1x**  |

### Interpretation

- v6 reduces overhead from ~50ms → ~3ms (initial clustering, p-values, mask
  conversion are now negligible relative to the permutation loop).
- Per-perm cost is stable at ~0.56ms. At this level, remaining costs are
  dominated by the Numba union-find edge iteration and threshold comparison.
- The cumulative speedup is **27.1x** vs the original implementation.
- At 5000 perms (typical research workload), expected time:
  0.003s + 5000 × 0.00056s = **2.8s** (vs ~82s original = **29x speedup**).
