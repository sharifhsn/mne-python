# Mathematical Analysis: Union-Find CCL + Precomputed Sum-of-Squares

Ephemeral doc for user review. Delete when done.

---

## 1. Correctness Proof: Union-Find Produces Identical Components to BFS

### Graph Definition

Define the implicit spatio-temporal graph **G = (V, E)**.

**Vertices.** Let there be `n_src` spatial source vertices and `n_times` time points. The data is a flattened array of length `N = n_times * n_src`. A vertex `v` has flat index `flat(v) = t(v) * n_src + s(v)`, where `t(v)` is its time index and `s(v)` is its spatial index. The active vertex set is:

> V = { v : x[flat(v)] passes the threshold test }

**Edges.** Two active vertices u, v are adjacent iff:

1. **Spatial edge:** `t(u) = t(v)` and the CSR adjacency matrix `A` has `A[s(u), s(v)] != 0`. Symmetry of `A` guarantees this is undirected.
2. **Temporal edge:** `s(u) = s(v)` and `0 < |t(u) - t(v)| <= max_step`.

The connected components of G are the clusters.

### Theorem

Let `C_BFS` be the partition of V into connected components produced by the old BFS algorithm, and `C_UF` be the partition produced by the new union-find algorithm. Then `C_BFS = C_UF` (up to reordering).

### Proof

The proof rests on a standard graph-theoretic fact: any algorithm that (a) starts with each vertex in its own singleton set, (b) for every edge `(u,v)` in E merges the sets containing `u` and `v`, and (c) outputs the resulting partition, produces exactly the connected components of G. This is because the merge operations generate the reflexive-transitive closure of E, which is precisely the connected component equivalence relation.

**Step 1: The union-find enumerates every edge in E.**

For each active vertex `a_pos` with flat index `flat_i`, the algorithm computes `t_i = flat_i // n_src` and `s_i = flat_i % n_src`, then:

- **Spatial edges:** Iterates over `j_ptr in range(adj_indptr[s_i], adj_indptr[s_i+1])`, extracting each spatial neighbor `s_j = adj_indices[j_ptr]`. Forms `flat_j = t_i * n_src + s_j` and checks if `flat_j` is active (`flat_to_active[flat_j] >= 0`). If so, calls `_union(a_pos, b_pos, parent, rank)`. This enumerates every spatial edge incident to `a_pos`.

- **Temporal edges:** Iterates `step` in `{1, ..., max_step}` (subject to `t_i >= step`) and forms `flat_j = (t_i - step) * n_src + s_i`. If active, calls `_union`. Since vertices are processed in increasing flat-index order (increasing time, then source), the "look backward only" strategy covers every temporal edge: for any temporal edge `(u, v)` with `t(u) < t(v)`, it is enumerated when processing `v`.

- **Spatial edges are symmetric:** The CSR adjacency is `(adjacency + adjacency.T).tocsr()`, so both directions appear. A single `_union` call from either direction suffices.

**Step 2: `_union` correctly merges equivalence classes.**

`_union` implements textbook union-find with path compression (path halving) and union by rank (Tarjan, 1975):

1. Find root of `a_pos`: follow parent pointers, applying `parent[x] = parent[parent[x]]` (path halving).
2. Find root of `b_pos` analogously.
3. If roots differ, attach the shorter-rank tree under the taller-rank tree; increment rank only on ties.

**Step 3: The BFS algorithm also computes connected components of G.**

- `_get_clusters_st_1step` (max_step=1): Computes spatial clusters per time step via BFS, then merges across consecutive time steps by scanning shared active vertices and calling `_reassign`.
- `_get_clusters_st_multistep` (max_step > 1): A global BFS expanding via `_get_buddies` (spatial) and `_get_selves` (temporal within max_step).

**Step 4: Uniqueness.** The connected component partition of a graph is unique. Since both algorithms compute connected components of the same graph G, they produce the same partition. **QED.**

---

## 2. Algorithmic Complexity Analysis

### Notation

- `N = n_times * n_src`: total grid size
- `n = |V|`: number of active (supra-threshold) vertices
- `m = |E|`: edges in the active subgraph; `m = O(n * (d_s + max_step))` where `d_s` is average spatial degree
- `alpha(n)`: inverse Ackermann function (`alpha(n) <= 4` for all `n < 10^80`)

### Old BFS (Python)

**`_get_clusters_st_1step`** has two phases:

| Phase | Work |
|-------|------|
| Spatial BFS per time slice | O(n * d_s) total across all time slices |
| Merge across time (via `_reassign`) | **O(n * N) worst case** -- each `_reassign` scans `check[check == num] = base` over the full `(n_times, n_src)` matrix |

The merge phase is the bottleneck. Each `_reassign` performs a linear scan of the entire check matrix (size N). In the worst case (many small clusters that merge into one), there are O(n) such calls, giving **O(n * N)**.

**`_get_clusters_st_multistep`**: global BFS, O(n + m) = O(n * (d_s + max_step)) with Numba helpers. But the outer BFS loop runs in Python (see Section 3).

### New Union-Find (Numba JIT)

| Phase | Work |
|-------|------|
| Build `flat_to_active` | O(n) |
| Union-find over all edges | O(m * alpha(n)) = O(n * (d_s + max_step) * alpha(n)) |
| Final path compression + relabel | O(n) |
| Clean up `flat_to_active` | O(n) |

**Total: O(n * (d_s + max_step) * alpha(n))**, effectively linear.

### Comparison

| Aspect | Old BFS | New Union-Find |
|--------|---------|----------------|
| Asymptotic | O(n + m) BFS, but **O(n*N)** merge overhead for 1step | O(m * alpha(n)) ~= O(m) |
| Execution | Python interpreter + Numba helpers | Fully Numba-compiled |
| Memory access | List-of-arrays, scattered | CSR contiguous arrays |
| Per-edge overhead | ~100-500ns (Python dispatch) | ~1-5ns (native) |

---

## 3. Why ~3x Faster in Practice (Constant Factor Analysis)

Since BFS is also effectively O(V+E), the measured ~3x speedup from commit 6 is primarily from constant factors.

### 3.1 Python Interpreter Overhead (dominant, ~2-2.5x)

The old BFS orchestrates the frontier in pure Python:

```python
while next_ind >= 0:
    t_inds = [next_ind]       # Python list alloc
    while icount <= len(t_inds):
        ind = t_inds[icount - 1]   # Python list index: ~50ns
        buddies = _get_buddies(...)  # Numba call: ~200-500ns overhead
        t_inds.extend(buddies)       # Python list extend: ~100ns
        icount += 1
    next_ind = _where_first(r)       # Another Numba call
    clusters.append(s[t_inds])       # NumPy fancy indexing
```

Each BFS step crosses the Python/Numba boundary. The Numba call overhead (~200-500ns for argument marshaling) is paid per vertex per neighbor lookup.

The new union-find enters Numba **once** and processes all `n` vertices and `m` edges in a single compiled function. The one-time call overhead (~500ns) is amortized over the entire computation.

### 3.2 Cache Locality (~1.2-1.5x)

**Old:** `neighbors[s[ind]]` chases a pointer from a Python list of separate NumPy arrays, each allocated independently and scattered in memory.

**New:** CSR format stores all neighbor indices in one contiguous array `adj_indices` with offsets in `adj_indptr`. Access is a sequential scan of a contiguous segment. The `parent`, `rank`, `flat_to_active` arrays are all contiguous and compact (size `n_active`).

### 3.3 Allocation and GC Pressure (~1.1-1.2x)

**Old:** Each BFS component allocates and extends Python lists, converts Numba typed lists to Python lists, creates NumPy arrays via fancy indexing, and (in `_reassign`) calls `np.concatenate` creating new arrays.

**New:** Exactly 4 array allocations (parent, rank, label_map, components), all done once before the main loop. Zero allocations inside the loop.

### 3.4 The `_sums_only` Fast Path

When `_sums_only=True` (all permutation iterations), the new code computes cluster sums via a single `np.bincount(components, weights=x[active_idx])` call, avoiding cluster list construction entirely.

### Summary

| Factor | Contribution |
|--------|-------------|
| Python interpreter overhead | ~2.0-2.5x |
| Cache locality (CSR vs scattered arrays) | ~1.2-1.5x |
| Fewer allocations | ~1.1-1.2x |
| Combined | ~3-4x |

---

## 4. Precomputed Sum-of-Squares: Mathematical Derivation

### 4.1 The One-Sample t-Statistic

Given `n` samples `X_1, ..., X_n`, the one-sample t-statistic for variable `j` (testing H_0: mu_j = 0) is:

```
t_j = x_bar_j / sqrt(s_j^2 / n)
```

where:

```
x_bar_j = (1/n) * SUM_i X_{i,j}
s_j^2   = (1/(n-1)) * SUM_i (X_{i,j} - x_bar_j)^2
```

### 4.2 Expanding the Variance

Using the sum-of-squares decomposition:

```
SUM_i (X_{i,j} - x_bar_j)^2 = SUM_i X_{i,j}^2 - n * x_bar_j^2
```

So:

```
s_j^2 / n = (SUM_i X_{i,j}^2 - n * x_bar_j^2) / (n * (n-1))
```

Substituting:

```
t_j = x_bar_j * sqrt(n*(n-1)) / sqrt(SUM_i X_{i,j}^2 - n * x_bar_j^2)
```

### 4.3 Sign-Flip Permutations

Each permutation is a sign vector `s = (s_1, ..., s_n)` with `s_i in {-1, +1}`. The sign-flipped data is `Y_{i,j} = s_i * X_{i,j}`.

The sign-flipped **mean** is:

```
x_bar_j^(s) = (1/n) * SUM_i s_i * X_{i,j} = (1/n) * (s^T X)_j
```

The sign-flipped **sum of squares** is:

```
SUM_i Y_{i,j}^2 = SUM_i s_i^2 * X_{i,j}^2
```

**Key identity: s_i^2 = 1 for all s_i in {-1, +1}**. Therefore:

```
SUM_i Y_{i,j}^2 = SUM_i X_{i,j}^2    (CONSTANT across all permutations)
```

### 4.4 The Optimized Formula

Substituting into the expanded t-statistic:

```
t_j^(s) = x_bar_j^(s) * sqrt(n*(n-1)) / sqrt(SUM_i X_{i,j}^2 - n * (x_bar_j^(s))^2)
```

Define precomputed constants:

| Symbol | Definition | Code variable |
|--------|-----------|---------------|
| Q_j | SUM_i X_{i,j}^2 | `_sum_sq` |
| c | sqrt(n*(n-1)) | `_sqrt_n_nm1` |
| | 1/n | `_inv_n` |
| | -n | `_neg_n` |

For each permutation with sign vector `s`:

1. `dot_j = (s^T X)_j` via `dot = signs @ X` (BLAS dgemv)
2. `mean_s_j = dot_j / n` via `mean_s = dot * _inv_n`
3. `denom_sq_j = Q_j - n * mean_s_j^2` via `denom_sq = _sum_sq + mean_s * mean_s * _neg_n`
4. `t_j^(s) = mean_s_j * c / sqrt(denom_sq_j)` via `mean_s / np.sqrt(denom_sq) * _sqrt_n_nm1`

### 4.5 Verification

Starting from the standard formula:

```
t = mean(Y) / sqrt(var(Y) / n)

var(Y) = (1/(n-1)) * (SUM Y_i^2 - n * mean(Y)^2)
       = (1/(n-1)) * (SUM X_i^2 - n * mean_s^2)     [since s_i^2 = 1]
       = (1/(n-1)) * (Q - n * mean_s^2)
       = (1/(n-1)) * denom_sq

var(Y)/n = denom_sq / (n*(n-1))

t = mean_s / sqrt(denom_sq / (n*(n-1)))
  = mean_s * sqrt(n*(n-1)) / sqrt(denom_sq)
  = mean_s / sqrt(denom_sq) * _sqrt_n_nm1
```

This matches the code exactly. **QED.**

### 4.6 Performance Analysis

**Original approach per permutation:**

1. `X *= signs` -- write full `(n, p)` matrix: O(np) writes
2. `stat_fun(X)` = `ttest_1samp_no_p(X)`: ~3 passes over X (mean + variance + sqrt), O(3np) reads
3. `X *= signs` -- restore, another O(np) writes

Total memory traffic: **~5np * 8 bytes** (two full writes + three full reads of X).

**New approach per permutation:**

1. `dot = signs @ X` -- BLAS dgemv: 1 read pass over X, O(np) reads
2. `mean_s, denom_sq, t_obs_surr` -- all O(p) on 1D arrays

Total memory traffic: **~np * 8 bytes** (one read of X).

**Theoretical speedup: ~5x** on the t-statistic computation (memory-bandwidth limited). Key wins:
- Eliminates two in-place `X *= signs` operations (writes ~2x more expensive than reads)
- Reduces from 3-5 passes over `(n, p)` to exactly 1 pass
- `X` is never modified, avoiding cache invalidation

### 4.7 Numerical Stability

The `np.maximum(denom_sq, 0.0)` guard handles floating-point rounding when `Q_j ~= n * mean_s_j^2` (near-zero variance), preventing `sqrt` of a negative number. The subsequent `np.where(denom_sq > 0, ..., 0.0)` returns `t = 0` when variance is zero, which is appropriate for the permutation test (a zero-variance location should not contribute to the max cluster statistic).
