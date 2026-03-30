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

---

## 2026-03-08: Spatio-temporal Numba union-find for neighbor-list path (v7)

### What changed

Added `_st_fused_ccl` — a Numba JIT function that replaces the Python BFS in
`_get_clusters_st` for the spatio-temporal neighbor-list adjacency path. This
is the **actual hot path** used by `spatio_temporal_cluster_1samp_test` when
the spatial adjacency is smaller than the number of tests (the common case:
20,484 spatial vertices × 15 timepoints = 307,260 tests).

The key discovery was that **all v1-v6 optimizations only helped the sparse
adjacency path** (`_get_components` / `_fused_ccl`), which is NOT what the
standard spatio-temporal benchmark uses. When `_setup_adjacency` detects
`adjacency.shape[0] != n_tests`, it converts to neighbor lists and uses
`_get_clusters_st` (Python BFS via Numba-decorated `_get_buddies` and
`_reassign`). The `_sums_only` fast path (v4) explicitly excluded list
adjacency with the condition `sparse.issparse(adjacency) or adjacency is False`.

**What the optimization does:**

1. **`_st_fused_ccl` Numba JIT**: Union-find over both spatial neighbors (from
   CSR adjacency) and temporal neighbors (same vertex at t±step). Accepts
   pre-identified active indices from `np.where(x_in)` to avoid O(n_total)
   scans — only touches the ~231 active vertices (0.08% of 307K total).

2. **`_use_st_fast` branch in `_do_1samp_permutations`**: Before the loop,
   converts neighbor lists to CSR format once. Inside the loop, bypasses
   `_find_clusters` entirely: threshold → `np.where` → `_st_fused_ccl` →
   `np.bincount` for cluster sums.

3. **Bug fix**: Also fixes a subtle bug in `_get_clusters_st_1step` where
   `_reassign` modifies `check1` in place during the nexts loop, but `check1_d`
   was captured as a snapshot before modifications. This causes stale values to
   orphan clusters that should be merged. Verified with BFS ground truth:
   union-find produces 55 clusters vs original's 57 (over-count).

### Why

After v6, profiling the actual `spatio_temporal_cluster_1samp_test` benchmark
revealed that `_get_clusters_st` (the neighbor-list BFS) was the dominant
per-permutation cost, consuming ~50-70% of per-perm time. None of the v1-v6
optimizations (reindexing, sums_only, fused_ccl) applied to this path because
they required sparse adjacency.

### Benchmark results (7/7 parity tests passed, 49/49 MNE unit tests passed)

#### CCL-only: _st_fused_ccl vs _get_clusters_st (real fsaverage, 50 perms)

| Path | Per-perm | Speedup |
|------|----------|---------|
| Original (_get_clusters_st BFS) | 0.88 ms | 1x |
| **_st_fused_ccl (Numba UF)** | **0.14 ms** | **6.2x** |

#### Synthetic graph (200×100 grid, 15 times, 300K vertices, 13% active)

| Path | Per-call | Speedup |
|------|----------|---------|
| Original (_get_clusters_st BFS) | 207 ms | 1x |
| **_st_fused_ccl (Numba UF)** | **1.28 ms** | **162x** |

#### Full permutation loop: _do_1samp_permutations (fsaverage, 1024 perms)

| Path | Per-perm | Speedup |
|------|----------|---------|
| Old (v6 st: _find_clusters → _get_clusters_st) | 1.67 ms | 1x |
| **New (v7: _st_fused_ccl + bypass _find_clusters)** | **0.97 ms** | **1.7x** |

Note: End-to-end improvement is 1.7x (not 6.2x) because the ttest computation
(0.73 ms) now dominates per-perm time.

#### Per-permutation breakdown (v7, 1024 perms, Numba JIT)

| Component | Per-perm (ms) | % |
|-----------|---------------|---|
| ttest (signs @ X) | 0.735 | 74.3% |
| np.where | 0.114 | 11.5% |
| threshold (x > t) | 0.109 | 11.0% |
| _st_fused_ccl (UF) | 0.026 | 2.6% |
| bincount sums | 0.004 | 0.4% |
| **Total** | **0.989** | |

### Interpretation

- The CCL bottleneck has been **eliminated**: it's now 2.6% of per-perm time
  (26 µs) vs the previous ~50-70%. The ttest matrix-vector multiply (0.73 ms)
  is now the clear bottleneck at 74.3%.
- For the standard 7-subject fsaverage benchmark, only 127 permutations are
  possible (2^7 - 1), so total test time is dominated by fixed overhead (~60ms)
  plus 127 × 0.97ms ≈ 123ms loop = ~183ms total.
- The union-find also **fixes a clustering bug** in the original `_get_clusters_st`,
  where in-place mutation during cross-time-step merges caused over-counting.
  The fix is automatically applied when Numba is available (the original path is
  unchanged for backwards compatibility when Numba is not installed).
- Active vertices are typically very sparse (~231 of 307K = 0.08%). The
  `np.where` + pre-identified active indices design keeps the Numba function
  O(n_active + edges_between_active) instead of O(n_total).
- At 5000 perms (typical research workload): ~60ms overhead + 5000 × 0.97ms
  = **5.0s** for the full test. The ttest (signs @ X) is the next target.

---

## 2026-03-08: Fused Numba ttest + threshold index extraction (v8)

### What changed

Two optimizations targeting the ttest computation, which was 74.3% of per-perm
time after v7:

**1. `_fused_ttest` Numba JIT**: Replaces the numpy sequence of `np.dot` + 8
elementwise operations (divide, multiply, add, maximum, sqrt, divide, multiply)
with a single Numba `prange` loop that reads each row of `X_T` once and writes
`t_buf` once. This eliminates 8 intermediate memory passes over 307K-element
arrays (~39 MB of memory traffic reduced to ~20 MB).

The function takes a transposed data matrix `X_T` (n_vars × n_samp, contiguous
for row-major access). For n_samp=7, the inner loop is just 7 multiply-adds
plus ~10 scalar ops per variable — entirely compute-bound per thread, with
multi-core parallelism via `prange`.

Falls back to serial `range` when Numba is not available (via the existing
`prange = range` fallback in `mne.fixes`).

**2. `_threshold_to_indices` and `_threshold_to_indices_neg` Numba JIT**:
Replace `np.where(t_buf > threshold)[0].astype(np.intp)` with a single Numba
loop that scans `t_buf` and writes matching indices directly into a
pre-allocated buffer. Avoids: (a) boolean array allocation (307K bytes),
(b) `np.where` Python overhead (tuple wrapping, array allocation, scan),
(c) `.astype(np.intp)` copy.

### Why

After v7, profiling showed the per-permutation breakdown was:

| Component | Time | % |
|-----------|------|---|
| ttest (dot + 8 elementwise) | 735 µs | 74.3% |
| np.where | 114 µs | 11.5% |
| threshold comparison | 109 µs | 11.0% |
| _st_fused_ccl (UF) | 26 µs | 2.6% |
| bincount sums | 4 µs | 0.4% |

The numpy ttest does `np.dot(signs, X)` (BLAS GEMV) followed by 8 separate
elementwise operations, each reading and writing 307K float64 values. This
produces ~39 MB of memory traffic per permutation. The fused approach reads
X_T (17.2 MB) once and writes t_buf (2.5 MB) once — a 2x reduction in memory
traffic, with all intermediate values kept in registers.

The `np.where` overhead (Python function call, boolean array allocation, index
extraction, tuple wrapping) was 114 µs — more than the CCL computation itself.
A Numba serial scan avoids all of this.

### Benchmark results (49/49 MNE unit tests passed, 7/7 parity tests passed)

#### Isolated ttest comparison (307K tests, 1024 perms)

| Component | v7 (µs) | v8 (µs) | Speedup |
|-----------|---------|---------|---------|
| ttest (dot+elem) | 735 | 250 | **2.94x** |
| thresh+indices | 351 | 258 | **1.36x** |
| CCL (UF) | 670 | 602 | 1.1x |
| bincount | 62 | 52 | 1.2x |
| **Total** | **1,820** | **1,163** | **1.56x** |

Note: CCL times are high in this benchmark because random data has ~14% active
vertices. With real data (~0.08% active), CCL is ~26 µs and the ttest savings
dominate.

#### End-to-end via `_do_1samp_permutations` (2048 perms)

| Version | Time | Per-perm | Speedup |
|---------|------|----------|---------|
| v7 (numpy ttest + np.where) | 4.31s | 2.10 ms | 1x |
| **v8 (fused ttest + Numba indices)** | **3.26s** | **1.59 ms** | **1.32x** |

Linear regression: overhead = -9ms (warm), per_perm = 1.42 ms.

#### Sub-breakdown of elementwise chain replaced (307K tests, 2000 perms)

| Operation | Time (µs) | % of chain |
|-----------|-----------|------------|
| mean_s /= n | 50 | 10.0% |
| mean_s² → denom | 47 | 9.5% |
| denom *= -n | 40 | 8.0% |
| denom += sum_sq | 57 | 11.4% |
| maximum(denom, 0) | 106 | 21.4% |
| sqrt(denom) | 84 | 17.0% |
| mean_s / denom | 71 | 14.2% |
| t_buf *= scale | 42 | 8.5% |
| **Total chain** | **496** | |

Each operation reads+writes 2.5 MB (307K × 8 bytes). The fused version
eliminates all 8 passes, keeping intermediate values in registers.

### Interpretation

- The fused Numba ttest is **2.94x faster** than the numpy dot+elementwise
  chain. The speedup comes from eliminating 8 intermediate memory passes,
  reducing memory traffic from ~39 MB to ~20 MB per permutation.
- `_threshold_to_indices` saves ~93 µs per permutation over `np.where` by
  avoiding boolean array allocation and Python function call overhead.
- Total per-permutation savings: ~657 µs (1,820 → 1,163 µs) in isolated
  breakdown, 1.32x in the full `_do_1samp_permutations` loop.
- The CCL union-find (51.7% of v8 time) appears dominant in this benchmark
  because of the high active fraction (14%). With real data (0.08% active),
  the CCL drops to ~26 µs and the fused ttest becomes the new bottleneck.
- Expected per-perm for real sparse data: ~250 (ttest) + ~258 (thresh+idx) +
  ~26 (CCL) + ~4 (bincount) ≈ **538 µs**, vs v7's ~989 µs = **1.84x**.
- At 5000 perms: ~60ms overhead + 5000 × 0.54ms = **2.7s** (vs v7's 5.0s).

### HPC Benchmark (AWS Batch c7a.4xlarge, 16 vCPU AMD EPYC 7R13)

Thermally stable results on dedicated EC2 Spot instance (no neighbor noise,
no thermal throttling). Python 3.12.13, NumPy 2.4.2, SciPy 1.17.1, Numba 0.64.0.

#### End-to-end (2048 perms, 5 runs, median)

| Version | Time | Per-perm | Speedup |
|---------|------|----------|---------|
| v7 (numpy ttest + np.where) | 6.350s | 3.101 ms | 1x |
| **v8 (fused ttest + Numba indices)** | **4.107s** | **2.005 ms** | **1.55x** |

Linear regression (v8): overhead = 9.5ms, per_perm = 1.994 ms.

Run-to-run variance (v8): 4.087 / 4.091 / 4.107 / 4.130 / 4.144s
(range: 57ms, CV < 0.7% — excellent stability).

#### Per-component breakdown (1024 perms)

| Component | v7 (µs) | v7 % | v8 (µs) | v8 % | Speedup |
|-----------|---------|------|---------|------|---------|
| signs | 5.8 | 0.2% | 6.7 | 0.3% | — |
| ttest (dot+elem / fused) | 1,309.4 | 41.8% | 505.4 | 24.9% | **2.59x** |
| thresh+indices | 431.3 | 13.8% | 412.8 | 20.4% | 1.04x |
| CCL (UF) | 1,299.9 | 41.5% | 1,030.5 | 50.8% | 1.26x |
| bincount | 85.6 | 2.7% | 73.1 | 3.6% | 1.17x |
| **Total** | **3,132.0** | | **2,028.6** | | **1.54x** |

#### HPC vs local comparison

| Metric | Local (M3 Mac) | HPC (EPYC 7R13) | Ratio |
|--------|-----------------|------------------|-------|
| v8 per-perm | 1.59 ms | 2.005 ms | 0.79x |
| v7 per-perm | 2.10 ms | 3.101 ms | 0.68x |
| v8/v7 speedup | 1.32x | 1.55x | — |
| Ttest speedup | 2.94x | 2.59x | — |
| Thresh+idx speedup | 1.36x | 1.04x | — |

The EPYC runs slower in absolute terms (fewer GHz, no Apple-silicon-specific
memory bandwidth advantages) but shows a **larger v8/v7 speedup ratio** (1.55x
vs 1.32x). This is because:
1. EPYC has lower per-core bandwidth, making the memory traffic reduction from
   fusing 8 elementwise ops into one pass more impactful.
2. The numpy threshold + np.where path is relatively slower on EPYC (431 µs vs
   351 µs on Mac), but `_threshold_to_indices` shows negligible improvement on
   EPYC (1.04x vs 1.36x on Mac) — likely because Numba's serial scan doesn't
   benefit from EPYC's wide vector units the way the numpy vectorized comparison
   does on Apple Silicon.

#### Projected production performance (EPYC, inner loop only)

At 5000 perms (typical research workload):
- v8: ~10ms overhead + 5000 × 2.0ms = **10.0s**
- v7: ~10ms overhead + 5000 × 3.1ms = **15.5s**
- Savings: **~5.5 seconds per test** (35% faster)

### Full End-to-End HPC Benchmark (AWS Batch, AMD EPYC 9R14, 16 vCPU)

Tests the actual researcher API `spatio_temporal_cluster_1samp_test` with
realistic parameters. v7 simulated via monkeypatching `_fused_ttest`,
`_threshold_to_indices`, `_threshold_to_indices_neg` with numpy equivalents.

**Setup**: 15 subjects × 20,484 vertices (fsaverage ico-5) × 15 timepoints
= 307,260 tests. Signal injected: 100 vertices × 5 timepoints at +2.0 SD.
`out_type="indices"` to keep memory manageable (~13,854 clusters).

Python 3.12.13, NumPy 2.4.2, SciPy 1.17.1, Numba 0.64.0.

#### Main timing (2048 perms, 5 runs, median)

| Version | Time | Per-perm | Speedup |
|---------|------|----------|---------|
| v7 (numpy ttest + np.where) | 4.901s | 2.393 ms | 1x |
| **v8 (fused ttest + Numba indices)** | **3.281s** | **1.602 ms** | **1.49x** |

Run-to-run variance (v8): 3.315 / 3.281 / 3.281 / 3.282 / 3.281s
(CV < 0.4% — excellent stability).

#### Parity verification

| Check | Result |
|-------|--------|
| t_obs match (atol=1e-10) | PASS |
| H0 match (atol=1e-10) | PASS |
| p-values match (atol=1e-10) | PASS |
| n_clusters match | PASS (13,854 vs 13,854) |
| Determinism (same seed) | PASS |
| **Overall** | **ALL PASS** |

#### Multi-tail parity + performance (2048 perms)

| Tail | Threshold | v8 | v7 | Speedup | Parity |
|------|-----------|-----|-----|---------|--------|
| 1 | 1.67 | 3.281s | 4.901s | 1.49x | ALL PASS |
| 0 | 1.67 | 5.961s | 7.445s | 1.25x | ALL PASS |
| -1 | -1.67 | 3.268s | 4.918s | 1.51x | ALL PASS |

Tail=0 runs both positive and negative tails, doubling work — the optimization
applies to both, yielding 1.25x (vs 1.49x-1.51x for single-tail).

#### Scaling (tail=1)

| Perms | v8 | v7 | Speedup | v8 ms/perm | v7 ms/perm |
|-------|-----|-----|---------|------------|------------|
| 256 | 0.657s | 0.873s | 1.33x | 2.566 | 3.412 |
| 512 | 1.062s | 1.469s | 1.38x | 2.075 | 2.869 |
| 1024 | 1.803s | 2.612s | 1.45x | 1.760 | 2.551 |
| 2048 | 3.283s | 4.994s | 1.52x | 1.603 | 2.439 |
| 4096 | 6.276s | 9.492s | 1.51x | 1.532 | 2.317 |

As n_perms increases, the fixed overhead is amortized, and the speedup
converges to the per-perm ratio (~1.55x).

#### Linear regression

| Version | Overhead | Per-perm |
|---------|----------|----------|
| v8 | 282.8 ms | 1.471 ms |
| v7 | 288.5 ms | 2.278 ms |
| **Per-perm speedup** | | **1.55x** |

Fixed overhead is identical (~283-289 ms) for both — this is initial t-test,
adjacency construction, initial clustering, and p-value computation. The
per-perm speedup (1.55x) matches the inner-loop benchmark on the EPYC 7R13.

#### Projected production performance

At 5000 perms (typical research workload):
- **v8: 7.6s** (283ms overhead + 5000 × 1.47ms)
- **v7: 11.7s** (289ms overhead + 5000 × 2.28ms)
- **Savings: 4.0s (35% faster)**

At 10000 perms (high-resolution analysis):
- **v8: 15.0s** vs **v7: 23.1s** — savings: **8.1s (35% faster)**

#### Key insight: fixed overhead and Amdahl's law

With 7 subjects and 1024 perms, the permutation loop is negligible (~3ms)
compared to the ~540ms fixed overhead (adjacency construction, initial
clustering). The v8 speedup was invisible at that scale.

With 15 subjects and 2048+ perms, the permutation loop dominates (82% of
wall time at 2048 perms, 95% at 10000 perms), and the 1.55x per-perm
speedup translates to 1.49-1.51x end-to-end improvement.

---

## 2026-03-09: Batched fused ttest — read X_T once per batch of 32 perms (v9)

### What changed

Added `_batched_fused_ttest` — a Numba `prange` kernel that computes t-stats
for B=32 permutations in a single pass over X_T. The outer loop `prange(n_vars)`
parallelizes across variables; for each variable `j`, an inner loop iterates
over `n_batch` sign vectors, reusing `X_T[j, :]` from L1 cache instead of
re-reading it from DRAM for each permutation.

Restructured the permutation loop in `_do_1samp_permutations` from a flat
`for seed_idx, order in enumerate(orders)` to a two-level structure:
1. **Outer loop**: batches of 32 permutations, calling `_batched_fused_ttest`
   once per batch (reads X_T once for 32 perms)
2. **Inner loop**: per-permutation threshold + CCL + bincount (cannot be batched)

Also precomputes all sign vectors as a contiguous 2D array `_all_signs`
(n_perms × n_samp) before the loop, so batch slicing is a zero-copy view.

### Why

After v8, the fused ttest was 22-25% of per-perm time on HPC (505 µs on EPYC).
Each call reads X_T (36.8 MB = 20,484 × 15 × 8 bytes) from DRAM. The signs
vector is tiny (120 bytes). By batching 32 perms per call, X_T is read once
(~37 MB) and reused for all 32 sign vectors — reducing memory traffic from
~42 MB/perm to ~1.3 MB/perm (32x reduction for the ttest step).

The threshold/CCL/bincount steps depend on each perm's t-stats and cannot be
batched, so they remain in the inner per-perm loop.

### Benchmark results (49/49 MNE unit tests passed, all parity checks passed)

### HPC Benchmark (AWS Batch, AMD EPYC 7R13, 16 vCPU)

**Setup**: 15 subjects × 20,484 vertices (fsaverage ico-5) × 15 timepoints
= 307,260 tests. `out_type="indices"`, 2048 permutations.

Python 3.12.13, NumPy 2.4.2, SciPy 1.17.1, Numba 0.64.0.

#### Main timing (2048 perms, 5 runs, median)

| Version | Time | Per-perm | Speedup |
|---------|------|----------|---------|
| v8 (unbatched fused ttest) | 4.362s | 2.130 ms | 1x |
| **v9 (batched B=32)** | **3.422s** | **1.671 ms** | **1.27x** |

Run-to-run variance (v9): 3.423 / 3.454 / 3.420 / 3.422 / 3.414s
(range: 40ms, CV < 0.5% — excellent stability).

#### Parity verification

| Check | Result |
|-------|--------|
| t_obs match (atol=1e-10) | PASS |
| H0 match (atol=1e-10) | PASS |
| p-values match (atol=1e-10) | PASS |
| n_clusters match | PASS (13,854 vs 13,854) |
| Determinism (same seed) | PASS |
| **Overall** | **ALL PASS** |

#### Multi-tail parity + performance (2048 perms)

| Tail | Threshold | v9 | v8 | Speedup | Parity |
|------|-----------|-----|-----|---------|--------|
| 1 | 1.67 | 3.422s | 4.362s | 1.27x | ALL PASS |
| 0 | 1.67 | 6.259s | 7.230s | 1.16x | ALL PASS |
| -1 | -1.67 | 3.438s | 4.343s | 1.26x | ALL PASS |

#### Scaling (tail=1)

| Perms | v9 | v8 | Speedup | v9 ms/perm | v8 ms/perm |
|-------|-----|-----|---------|------------|------------|
| 256 | 0.745s | 0.880s | 1.18x | 2.910 | 3.439 |
| 512 | 1.123s | 1.357s | 1.21x | 2.193 | 2.651 |
| 1024 | 1.898s | 2.371s | 1.25x | 1.853 | 2.315 |
| 2048 | 3.413s | 4.356s | 1.28x | 1.667 | 2.127 |
| 4096 | 6.494s | 8.362s | 1.29x | 1.586 | 2.042 |

#### Linear regression

| Version | Overhead | Per-perm |
|---------|----------|----------|
| v9 (batched) | 356.6 ms | 1.497 ms |
| v8 (unbatched) | 362.6 ms | 1.952 ms |
| **Per-perm speedup** | | **1.30x** |

Fixed overhead is identical (~357-363 ms). The per-perm speedup (1.30x) comes
entirely from reduced memory traffic in the batched ttest.

#### Projected production performance

At 5000 perms (typical research workload):
- **v9: 7.8s** (357ms overhead + 5000 × 1.50ms)
- **v8: 10.1s** (363ms overhead + 5000 × 1.95ms)
- **Savings: 2.3s (23% faster)**

At 10000 perms (high-resolution analysis):
- **v9: 15.3s** vs **v8: 19.9s** — savings: **4.6s (23% faster)**

### Cumulative progress (full end-to-end, EPYC, 2048 perms)

| Version | Time | Per-perm | vs v7 |
|---------|------|----------|-------|
| v7 (numpy ttest + np.where) | 4.901s | 2.393 ms | 1x |
| v8 (fused ttest + Numba indices) | 3.281s | 1.602 ms | 1.49x |
| **v9 (batched B=32)** | **3.422s** | **1.671 ms** | **1.43x** |

Note: v9's median is slightly higher than v8's from the previous session due to
different EPYC instance (7R13 vs 9R14) and different run conditions. The correct
A/B comparison is v9 vs v8-unbatched on the **same instance** (3.422s vs 4.362s
= 1.27x), not across sessions.

### Interpretation

- The batched ttest reduces per-perm cost by 1.30x on EPYC by reading X_T
  (36.8 MB) once per 32 permutations instead of once per permutation.
- Memory traffic reduction: ~42 MB/perm → ~1.3 MB/perm for the ttest step.
- The benefit increases with n_perms as fixed overhead is amortized: 1.18x at
  256 perms → 1.29x at 4096 perms.
- Remaining per-perm breakdown (estimated from linear regression):
  - Batched ttest: ~455 µs saved (from ~505 to ~50 µs amortized per perm)
  - Threshold+indices: ~413 µs (unchanged)
  - CCL (UF): ~1,031 µs (unchanged, high due to random data)
  - bincount: ~73 µs (unchanged)
- Next optimization target: threshold+indices parallelization (22% of time),
  or full Numba inner loop to eliminate Python loop overhead.

---

## 2026-03-09: Parallel threshold extraction — attempted, reverted (regression)

### What was tried

Replaced serial `_threshold_to_indices` and `_threshold_to_indices_neg`
(`@jit()`, single-threaded scan) with parallel three-phase stream compaction
(`@jit(parallel=has_numba)`, count/prefix-sum/scatter with 64 chunks):

1. **Phase 1** (prange): each chunk counts elements passing threshold.
2. **Phase 2** (serial): exclusive prefix sum over 64 chunk counts → write offsets.
3. **Phase 3** (prange): each chunk scatters matching indices at its computed offset.

Each chunk writes to a non-overlapping region of the output buffer, so no
atomic operations are needed.

### Why it failed

The parallel approach is a **regression** at the realistic 5% supra-threshold
density (threshold=1.67 on random data). The serial scan is only ~270 µs for
307K elements, and the parallel version takes ~385 µs — **42% slower**.

Root causes:
- **Two prange barriers**: each costs ~30-50 µs on EPYC (thread sync). Total
  overhead: ~60-100 µs, which is 22-37% of the serial scan time.
- **Triple data pass**: serial does 1 pass (compare + write); parallel does
  count + prefix_sum + scatter = 3 passes, tripling cache traffic.
- **Small workload**: 307K × 8B = 2.4 MB fits in L3. The serial scan at
  ~270 µs is already 8.9 GB/s — bandwidth is not the bottleneck, instruction
  throughput is.

At higher densities (20-50%), the parallel version wins (1.7-2.8x) because
there's enough work to amortize the synchronization cost. But the realistic
density for statistical thresholds is 2-10%.

### HPC Results (AMD EPYC 7R13, 16 vCPU)

#### Micro-benchmark (isolated threshold function, 500 iterations)

| Density | Serial (µs) | Parallel (µs) | Speedup |
|---------|-------------|---------------|---------|
| 5% (thresh=1.65) | 271 | 385 | **0.70x** |
| 20% (thresh=0.84) | 743 | 426 | 1.76x |
| 50% (thresh=0.0) | 1413 | 508 | 2.77x |

All parity checks PASS (identical output arrays).

#### End-to-end (fsaverage ico-5, 2048 perms, tail=1)

| Config | Median | Per-perm |
|--------|--------|----------|
| v10 (parallel threshold) | 3.649s | 1.782 ms |
| v9 (serial threshold) | 3.451s | 1.685 ms |

**End-to-end: 0.95x (5% slower)**

#### Linear regression

| Config | Overhead | Per-perm |
|--------|----------|----------|
| v10 (parallel) | 360 ms | 1.607 ms |
| v9 (serial) | 384 ms | 1.508 ms |

**Per-perm: 0.94x (6% slower)**

### Decision

**Reverted.** The serial `_threshold_to_indices` (v8 implementation) is
already near-optimal for this workload. The threshold scan represents ~18%
of per-perm time, but parallelizing it at ~5% density adds more overhead
than it saves.

### What's left in the per-perm budget

Post-v9 per-perm breakdown (EPYC 7R13, 1.508ms/perm, random data):
- Batched ttest (amortized): ~50 µs (3%)
- Threshold scan (serial Numba): ~270 µs (18%)
- CCL union-find (serial Numba): ~600 µs (40%) [~26 µs with real data]
- bincount + argmax: ~73 µs (5%)
- Python loop overhead + NumPy ops: ~515 µs (34%)

Remaining optimization targets:
1. **Python loop overhead** (~515 µs, 34%): Move entire inner loop to Numba
   or restructure to minimize per-iteration Python overhead.
2. **CCL with random data** (~600 µs, 40%): Only dominant with random data;
   real data produces compact clusters (~26 µs). May not be worth optimizing
   since benchmark random data is worst-case, not representative.
3. **GPU pipeline**: Fuse all steps (ttest + threshold + CCL + reduce) on GPU
   to eliminate all Python loop overhead and data transfer.

---

## 2026-03-09: v10 — Fused Numba inner loop (`_perm_batch_fast`)

### What changed

Added `_perm_batch_fast` — a single `@jit()` function that fuses the entire
per-permutation inner loop: threshold scan → CCL (`_st_fused_ccl`) → weighted
bincount → argmax(abs). This replaces the Python `for b in range(n_batch)` loop
that previously called individual Numba kernels from Python.

The function handles both tail directions (positive and negative thresholds) in
a single compiled function, eliminating Python interpreter overhead, NumPy
function dispatch, and temporary array allocations per permutation.

### Why

Post-v9 profiling on EPYC 7R13 showed ~515 µs/perm (34%) was spent in Python
loop overhead — the interpreter iterating the per-perm loop, calling Numba
kernels, and running NumPy bincount/argmax. By moving the entire loop body into
Numba, all of this becomes compiled native code.

### Bug fix: tail=-1 threshold

The initial implementation had a bug in the negative threshold computation.
For `tail=-1`, the threshold is already negative (e.g., -1.67), but the code
was computing `neg_thresh = -threshold = 1.67`, which selected almost all
elements. Fixed to: `neg_thresh = -threshold if tail == 0 else threshold`.

### HPC results (AMD EPYC 7R13, 16 vCPU, 307K tests, random data)

**End-to-end (2048 perms, tail=1):**

| Version | Median | ms/perm |
|---------|--------|---------|
| v10 (fused Numba) | 3.243s | 1.584 |
| v9 (Python loop)  | 3.443s | 1.681 |

**Speedup: 1.06x end-to-end**

**Linear regression (slope = marginal per-perm cost):**

| Version | Overhead | Per-perm slope |
|---------|----------|----------------|
| v10 | 370.4 ms | 1.403 ms |
| v9  | 351.2 ms | 1.509 ms |

**Per-perm slope speedup: 1.08x** (savings: ~106 µs/perm)

**Multi-tail results (2048 perms):**

| Tail | v10 | v9 | Speedup |
|------|-----|-----|---------|
| tail=1  | 3.243s | 3.443s | 1.06x |
| tail=0  | 5.847s | 6.250s | 1.07x |
| tail=-1 | 3.195s | 3.426s | 1.07x |

**Scaling (tail=1):**

| Perms | v10 | v9 | Speedup |
|-------|-----|-----|---------|
| 256   | 0.716s | 0.749s | 1.05x |
| 512   | 1.088s | 1.125s | 1.03x |
| 1024  | 1.798s | 1.900s | 1.06x |
| 2048  | 3.244s | 3.450s | 1.06x |
| 4096  | 6.145s | 6.505s | 1.06x |

Parity: ALL PASS (t_obs, H0, p-values, n_clusters match exactly for all tails).

### Analysis

The 106 µs/perm saving is less than the 515 µs estimated Python overhead.
This is because:

1. **Numba overhead replaces Python overhead**: The `_perm_batch_fast` function
   still has per-iteration costs (function dispatch, buffer management) even in
   compiled code. The savings come from eliminating Python interpreter overhead
   and NumPy function call overhead, but the compute work is the same.

2. **CCL dominates**: With random data, CCL takes ~600 µs/perm (40%). The Python
   overhead of ~515 µs was estimated by difference (total - instrumented Numba
   steps). Some of that "overhead" may have been memory system effects (cache
   misses between Numba calls) rather than pure Python interpreter time.

3. **Consistent across scales**: The ~6% speedup is stable across 256-4096 perms,
   confirming it's a genuine per-perm improvement, not a one-time overhead.

### Cumulative speedup vs original MNE

For the standard benchmark (fsaverage ico-5, 15 subjects, 307K tests, 2048 perms):

| Version | ms/perm | vs v0 | vs prev |
|---------|---------|-------|---------|
| v0 (original MNE) | ~6.3 | 1.0x | — |
| v5 (Numba UF + precomp ttest) | ~3.1 | 2.0x | — |
| v7 (ST Numba UF) | ~2.3 | 2.7x | — |
| v8 (fused ttest + threshold) | ~1.51 | 4.2x | 1.56x |
| v9 (batched ttest) | ~1.51 | 4.2x | 1.0x |
| v10 (fused inner loop) | ~1.40 | 4.5x | 1.08x |

### What's left in the per-perm budget

Post-v10 per-perm breakdown (EPYC 7R13, ~1.40ms/perm, random data):
- Batched ttest (amortized): ~50 µs (4%)
- Threshold+CCL+bincount+argmax (fused Numba): ~1,350 µs (96%)
  - Of which CCL: ~600 µs (43%) [~26 µs with real data]
  - Threshold scan: ~270 µs (19%)
  - bincount+argmax: ~73 µs (5%)
  - Remaining inter-call overhead: ~0 µs (eliminated)

The Python loop overhead has been eliminated. The remaining per-perm cost is
almost entirely the Numba compute kernels themselves. Further optimization
would require either:
1. **Algorithmic improvements** to CCL (e.g., parallel union-find)
2. **GPU pipeline** to exploit massive parallelism across permutations
3. **Real-data optimization**: With real data (not random), CCL drops from ~600
   to ~26 µs, making total per-perm ~0.85ms (7.4x vs original MNE).

---

## 2026-03-09: v11 — Parallel permutation processing (`prange(n_batch)`)

### What changed

Changed `_perm_batch_fast` from `@jit()` with `range(n_batch)` to
`@jit(parallel=has_numba)` with `prange(n_batch)`. Each permutation's inner
loop (threshold scan → CCL → weighted bincount → argmax) now runs on its own
core in parallel.

**Key implementation details:**
- Pre-allocated **2D work buffers**: `idx_bufs(B, n_vars)`, `flat_maps(B, n_vars)`,
  `sums_bufs(B, n_vars)` — each `prange` iteration indexes its own row `[b]`,
  eliminating data races.
- Memory cost: 32 × 307K × 24B = **235 MB** (3 arrays × 8B each).
- `_st_fused_ccl` Phase 4 resets only touched entries in `flat_map[b]` back to
  -1 after each call, so per-perm buffers stay clean between batches without
  full reinitialization.
- **Two-phase parallelism**: `_batched_fused_ttest` uses `prange(n_vars)`
  (parallel across 307K variables), then `_perm_batch_fast` uses `prange(n_batch)`
  (parallel across 32 perms). These run sequentially — no nested parallelism.

### Why

Post-v10, 96% of per-perm time was in the fused Numba inner loop running
**serially** on a single core. Meanwhile, the batched ttest (only 4% of time)
already used all 16 cores via `prange(n_vars)`. This meant 96% of wall time
used 1/16th of available compute. The per-perm CCL/threshold/bincount work is
completely independent across permutations — a natural `prange` target.

### v10 baseline method

v10 baseline simulated by `_serial_perm_batch_fast` — an identical `@jit()`
function that uses `range()` instead of `prange()`, monkeypatched into
`cluster_level` during the v10 benchmark run.

### HPC results (AMD EPYC 7R13, 16 vCPU, 307K tests, random data)

**End-to-end (2048 perms, tail=1):**

| Version | Median | ms/perm |
|---------|--------|---------|
| v11 (parallel prange) | 1.635s | 0.799 |
| v10 (serial range)    | 3.774s | 1.843 |

**Speedup: 2.31x end-to-end**

**Linear regression (slope = marginal per-perm cost):**

| Version | Overhead | Per-perm slope |
|---------|----------|----------------|
| v11 | 388.9 ms | 0.608 ms |
| v10 | 321.3 ms | 1.687 ms |

**Per-perm slope speedup: 2.77x**

**Multi-tail results (2048 perms):**

| Tail | v11 | v10 | Speedup |
|------|-----|-----|---------|
| tail=1  | 1.635s | 3.774s | 2.31x |
| tail=0  | 2.700s | 6.719s | 2.49x |
| tail=-1 | 1.620s | 3.651s | 2.25x |

tail=0 is faster because it processes both positive and negative thresholds per
perm, giving each `prange` iteration more work to do (better amortization of
thread scheduling overhead).

**Scaling (tail=1):**

| Perms | v11 | v10 | Speedup |
|-------|-----|-----|---------|
| 256   | 0.567s | 0.784s | 1.38x |
| 512   | 0.713s | 1.194s | 1.68x |
| 1024  | 1.015s | 2.025s | 2.00x |
| 2048  | 1.632s | 3.773s | 2.31x |
| 4096  | 2.869s | 7.268s | 2.54x |

Scaling improves with n_perms because more permutations per batch means more
parallel work per `prange` call. At 256 perms (8 batches of 32), there are
only 32 concurrent tasks for 16 cores. At 4096 perms (128 batches), each batch
fully utilizes all cores.

**Realistic data benchmark (dSPM source estimates, fsaverage ico-5):**

| Version | Median | Speedup |
|---------|--------|---------|
| v11 (parallel) | 0.848s | 1.46x |
| v10 (serial)   | 1.237s |  —    |

With real data, CCL is much faster (~26 µs vs ~600 µs with random data), so
the total per-perm work is smaller and there's less to parallelize. Still a
solid 1.46x improvement.

Parity: ALL PASS (t_obs, H0, p-values, n_clusters match exactly for all tails).

### Analysis

The 2.77x per-perm speedup on 16 vCPUs (not 16x) reflects several factors:

1. **Batch size = 32, cores = 16**: Each batch launches 32 tasks on 16 cores,
   giving 2x oversubscription. Numba's OpenMP runtime distributes these across
   cores, but each core handles ~2 perms sequentially.

2. **Work imbalance**: With random data at 5% density, most perms have similar
   supra-threshold counts, but CCL time varies with cluster structure. Some
   perms complete faster, leaving cores idle while others finish.

3. **Memory bandwidth**: 16 cores sharing L3 cache compete for bandwidth. The
   CCL's random-access pattern (union-find on sparse graph) is memory-latency
   bound, not compute-bound.

4. **Amdahl's law**: The batched ttest step (4% of time) remains sequential
   (parallel across vars, but serial across batches). With 96% parallelizable
   work and 16 cores, theoretical max is ~1/(0.04 + 0.96/16) = 10x. The 2.77x
   achieved is reasonable given memory bandwidth constraints.

### Cumulative speedup vs original MNE

For the standard benchmark (fsaverage ico-5, 15 subjects, 307K tests, 2048 perms):

| Version | ms/perm | vs v0 | vs prev |
|---------|---------|-------|---------|
| v0 (original MNE) | ~6.3 | 1.0x | — |
| v5 (Numba UF + precomp ttest) | ~3.1 | 2.0x | — |
| v7 (ST Numba UF) | ~2.3 | 2.7x | — |
| v8 (fused ttest + threshold) | ~1.51 | 4.2x | 1.56x |
| v9 (batched ttest) | ~1.51 | 4.2x | 1.0x |
| v10 (fused inner loop) | ~1.40 | 4.5x | 1.08x |
| **v11 (parallel perms)** | **~0.61** | **10.3x** | **2.77x** |

### What's left in the per-perm budget

Post-v11 per-perm breakdown (EPYC 7R13, ~0.61ms/perm wall time, random data):
- The per-perm compute work is unchanged from v10 (~1.40ms of single-core work).
- Wall time is ~0.61ms because 16 cores process ~2.3 perms' worth of work per
  wall-time millisecond.
- Further parallelism gains are limited by memory bandwidth and Amdahl's law.

Remaining optimization targets:
1. **GPU pipeline**: Fuse all steps (ttest + threshold + CCL + reduce) on GPU.
   GPU has vastly more cores (thousands vs 16) and higher memory bandwidth.
2. **Algorithmic CCL improvements**: Parallel union-find within a single perm
   (currently serial). Only matters for random data (~600 µs); real data CCL
   is already ~26 µs.
3. **With real data**: Per-perm wall time is likely ~0.41ms (7.4x original from
   single-core budget of ~0.85ms / 2.77x parallelism gain ≈ 0.31ms, plus
   overhead). The 1.46x measured on realistic data confirms this range.

---

## 2026-03-09: v12 — Fixed-overhead optimizations

### What changed

Three quick-win optimizations targeting the ~384ms fixed overhead (the constant
cost independent of n_perms):

1. **Vectorize `_pval_from_histogram`** (lines 1185-1207): Replaced
   O(n_clusters × n_perms) list comprehension with O(n_clusters × log(n_perms))
   binary search via `np.searchsorted`. Sort H0 once, then vectorized lookup.

2. **Skip `buffer_size` verification for built-in stat fns** (lines ~1790-1803):
   `ttest_1samp_no_p` and `f_oneway` are provably variable-independent.
   Added identity check `stat_fun is not ttest_1samp_no_p and stat_fun is not f_oneway`
   to skip the expensive 308-chunk re-computation loop.

3. **Vectorize sign pre-computation** (line ~1373): Replaced Python loop
   `for _i, _order in enumerate(orders): _all_signs[_i] = 2.0 * _order - 1.0`
   with one-liner `_all_signs = 2.0 * np.array(orders, dtype=np.float64) - 1.0`.

### Why

Post-v11, per-perm cost is 0.61ms (10.3x vs original). The fixed overhead
(~384ms on HPC) became a significant fraction of total runtime:
- At 2048 perms: 384ms / (384 + 2048×0.61) = 24% of total time
- At 256 perms: 384ms / (384 + 256×0.61) = 71% of total time

These three quick wins reduce that overhead with minimal code changes.

### HPC results (AMD EPYC 7R13, 16 vCPU)

#### Micro-benchmarks (individual optimizations)

| Optimization | Old | New | Speedup | Parity |
|---|---|---|---|---|
| `_pval_from_histogram` (tail=1) | 94.3 ms | 0.94 ms | **100x** | PASS |
| `_pval_from_histogram` (tail=0) | 110.2 ms | 0.95 ms | **116x** | PASS |
| `_pval_from_histogram` (tail=-1) | 94.0 ms | 0.95 ms | **99x** | PASS |
| `buffer_size` verification | 17.4 ms | 0 ms (skipped) | **∞** | N/A |
| Sign pre-computation | 4.88 ms | 0.48 ms | **10.1x** | PASS |

Total micro-benchmark savings: ~94 + 17 + 4.4 = **~116 ms**

#### End-to-end A/B benchmark (v12 vs v11)

| n_perms | v12 | v11 | Speedup | Parity |
|---|---|---|---|---|
| 256 | 0.425 s | 0.531 s | 1.251x | ALL PASS |
| 512 | 0.592 s | 0.673 s | 1.137x | ALL PASS |
| 1024 | 0.896 s | 0.979 s | 1.093x | ALL PASS |
| 2048 | 1.470 s | 1.562 s | 1.063x | ALL PASS |
| 4096 | 2.619 s | 2.737 s | 1.045x | ALL PASS |

**Note**: The A/B comparison only monkeypatches fix #1 (`_pval_from_histogram`).
Fixes #2 and #3 are measured separately in micro-benchmarks. The e2e speedup
reflects mostly fix #1's impact.

#### Linear regression

| Version | Overhead | Slope (ms/perm) |
|---|---|---|
| v12 | 298.6 ms | 0.568 ms/perm |
| v11 | 384.4 ms | 0.575 ms/perm |

**Overhead saved: 85.8 ms** (22.3% reduction in fixed overhead).
Per-perm slope unchanged (0.007 ms diff — confirms these are overhead-only fixes).

#### Multi-tail parity (2048 perms)

| Tail | t_obs | H0 | p-values | Overall |
|---|---|---|---|---|
| +1 | PASS | PASS | PASS | ALL PASS |
| 0 | PASS | PASS | PASS | ALL PASS |
| -1 | PASS | PASS | PASS | ALL PASS |

### Cumulative progress

| Version | Overhead (ms) | Per-perm (ms) | E2E @ 2048 | vs Original |
|---|---|---|---|---|
| Original MNE | ~880 | ~6.3 | ~13.8 s | 1.0x |
| v11 (parallel) | ~384 | ~0.61 | ~1.63 s | 8.5x |
| **v12 (overhead)** | **~299** | **~0.57** | **~1.47 s** | **9.4x** |

The overhead savings compound most at low perm counts:
- At 256 perms: v12 saves ~86ms out of ~540ms total = 16% wall-time reduction
- At 4096 perms: v12 saves ~86ms out of ~2700ms total = 3% wall-time reduction

### Files changed

- `mne/stats/cluster_level.py` — Three targeted edits (total ~20 lines changed)
- `gpu_accel/bench_v12_overhead.py` — New A/B benchmark script

---

## 2026-03-09: v13 — Fast initial clustering via `_st_fused_ccl`

### What changed

Added Numba union-find fast path in `_find_clusters_1dir` for spatio-temporal
list adjacency. When `has_numba` is True and `adjacency` is a list (the
standard spatio-temporal case), the initial t_obs clustering now uses
`_st_fused_ccl` (Numba union-find) instead of `_get_clusters_st` (Python BFS).

The initial clustering is called once before the permutation loop to find
clusters in the observed data. Previously it used the same Python BFS that
was already replaced inside the permutation loop (v7), but the initial
clustering path was never updated.

### Why

The initial clustering via Python BFS was the single largest remaining
overhead component (~145ms on HPC). It accounted for ~55% of the v12
residual overhead (263ms). The Numba union-find (`_st_fused_ccl`) was
already proven in the permutation loop across millions of calls — it just
needed to be wired up for the one-time initial clustering too.

### HPC results (AMD EPYC 9R14, 16 vCPU)

**Note**: This benchmark landed on EPYC 9R14 (newer gen) vs EPYC 7R13
for v12. The per-perm slopes are faster due to the newer CPU. Cross-version
overhead comparisons remain valid since overhead is dominated by Python/NumPy
which scales similarly.

#### Micro-benchmark: initial clustering

| Path | Time | Clusters | Speedup | Parity |
|---|---|---|---|---|
| Old (Python BFS) | 145.5 ms | 13,974 | — | — |
| New (Numba UF) | 18.1 ms | 13,974 | **8.0x** | PASS |

#### End-to-end A/B benchmark (v13 vs v12)

| n_perms | v13 | v12 | Speedup | Parity |
|---|---|---|---|---|
| 256 | 0.233 s | 0.357 s | 1.531x | ALL PASS |
| 512 | 0.329 s | 0.425 s | 1.291x | ALL PASS |
| 1024 | 0.499 s | 0.660 s | 1.324x | ALL PASS |
| 2048 | 0.905 s | 0.999 s | 1.104x | ALL PASS |
| 4096 | 1.646 s | 1.739 s | 1.057x | ALL PASS |

#### Linear regression

| Version | Overhead | Slope (ms/perm) |
|---|---|---|
| v13 | 136.4 ms | 0.369 ms/perm |
| v12 | 263.5 ms | 0.361 ms/perm |

**Overhead saved: 127.1 ms** (48.3% reduction in fixed overhead).
Per-perm slope unchanged (-0.008 ms diff — confirms this is overhead-only).

#### Multi-tail parity (2048 perms)

| Tail | t_obs | H0 | p-values | Overall |
|---|---|---|---|---|
| +1 | PASS | PASS | PASS | ALL PASS |
| 0 | PASS | PASS | PASS | ALL PASS |
| -1 | PASS | PASS | PASS | ALL PASS |

### Cumulative progress

| Version | Overhead (ms) | Per-perm (ms) | E2E @ 2048 | vs Original |
|---|---|---|---|---|
| Original MNE | ~880 | ~6.3 | ~13.8 s | 1.0x |
| v12 (overhead fixes) | ~264 | ~0.36 | ~1.00 s | 13.8x |
| **v13 (init CCL)** | **~136** | **~0.37** | **~0.89 s** | **15.5x** |

Note: The EPYC 9R14 per-perm slope (~0.37ms) is faster than the EPYC 7R13
(~0.57ms), so the e2e comparison reflects both the code optimization AND
the newer hardware. On the same EPYC 7R13, the v13 overhead savings alone
would yield ~1.47s - 0.127s = ~1.34s @ 2048 (10.3x vs original).

### Files changed

- `mne/stats/cluster_level.py` — Added `_st_fused_ccl` fast path in
  `_find_clusters_1dir` (lines 1129-1169)
- `gpu_accel/bench_v13_init_ccl.py` — New A/B benchmark script
- `gpu_accel/profile_overhead.py` — Overhead profiling script
